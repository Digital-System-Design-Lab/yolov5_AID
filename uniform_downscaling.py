# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from torch import Tensor
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

import torchjpeg.torchjpeg.src.torchjpeg.dct as TJ_dct#torchjpeg 에서 dct 가져오기
import torchjpeg.torchjpeg.src.torchjpeg.dct._color as TJ_ycbcr #추가
import torchjpeg.torchjpeg.src.torchjpeg.quantization.ijg as TJ_ijg #torchjpeg에서 quantization 가져오기
import torchjpeg.torchjpeg.src.torchjpeg.dct._block as TJ_block #추가
from einops import rearrange # pip install einops (텐서를 좀 더 자유자재로 쓸수있게함)
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from PIL import Image
import matplotlib.pyplot as plt

def simulate_jpeg_compression(image_tensor, quality=75):
    # Function to simulate JPEG compression and decompression for batches of images
    batch_size = image_tensor.shape[0]  # Get batch size
    compressed_images = []
    
    for i in range(batch_size):
        # Convert the tensor to PIL Image for each image in the batch
        pil_img = TF.to_pil_image(image_tensor[i].cpu())
        # Create an in-memory buffer
        buffer = io.BytesIO()
        # Save the image to the buffer as JPEG with specified quality
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        # Load the image back from the buffer
        compressed_img = Image.open(buffer)
        # Convert back to tensor
        image_tensor_compressed = TF.to_tensor(compressed_img).to(image_tensor.device)
        compressed_images.append(image_tensor_compressed)
    
    # Stack all compressed images back into a batch
    return torch.stack(compressed_images)

def visualize_and_save_images(original, compressed, title1='Original Image', title2='Compressed Image', filename_prefix='image'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Visualize original image
    axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title(title1)
    axs[0].axis('off')

    # Visualize compressed image
    axs[1].imshow(compressed.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title(title2)
    axs[1].axis('off')

    # Save the figure
    plt.savefig(f"{filename_prefix}_comparison.jpg")
    plt.show()

    # Save individual images
    original_image = original.permute(1, 2, 0).cpu().numpy()
    compressed_image = compressed.permute(1, 2, 0).cpu().numpy()

    # Ensure the images are in [0, 1] range for imsave
    original_image = np.clip(original_image, 0, 1)
    compressed_image = np.clip(compressed_image, 0, 1)

    # Save original image
    plt.imsave(f"{filename_prefix}_original.jpg", original_image)

    # Save compressed image
    plt.imsave(f"{filename_prefix}_compressed.jpg", compressed_image)
    
def save_one_txt(predn, save_conf, shape, file):
    """Saves one detection result to a txt file in normalized xywh format, optionally including confidence."""
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")

# YCbCr 값 확인 함수
def check_ycbcr_values(ycbcr_tensor: Tensor, data_range: float = 255):
    y_min, y_max = ycbcr_tensor[:, 0, :, :].min().item(), ycbcr_tensor[:, 0, :, :].max().item()
    cb_min, cb_max = ycbcr_tensor[:, 1, :, :].min().item(), ycbcr_tensor[:, 1, :, :].max().item()
    cr_min, cr_max = ycbcr_tensor[:, 2, :, :].min().item(), ycbcr_tensor[:, 2, :, :].max().item()

    print(f"Y range: {y_min:.2f} to {y_max:.2f}")
    print(f"Cb range: {cb_min:.2f} to {cb_max:.2f}")
    print(f"Cr range: {cr_min:.2f} to {cr_max:.2f}")
    
def save_one_json(predn, jdict, path, class_map):
    """
    Saves one JSON detection result with image ID, category ID, bounding box, and score.

    Example: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def delta_encode(coefs):#DCT 계수에 대해 / 델타 인코딩 수행(데이터 압축에서 자주 사용되는 기법)
    #델타 인코딩은 연속된 데이터 사이의 차이(델타)만 저장하는 방식
    #입력은 DCT 계수 텐서(coefs) // 각 블록의 dc 계수에 대해 델타 인코딩 적용 / ac는 그대로 유지
    #coefs 크기 (B, C, H*W/64, 64)
    #H*W/64가 블록갯수임
    ac = coefs[..., 1:]             # b 1 4096 63 #나머지는 AC 계수(63개)
    dc = coefs[..., 0:1]            # b 1 4096 1 #첫번째 요소는 DC계수(1개) 모든 블록에서 DC값을 추출
    dc = torch.cat([dc[..., 0:1, :], dc[..., 1:, :] - dc[..., :-1, :]], dim=-2)#각 DC 계수에서 바로 이전 DC 계수를 빼는 연산 
    #첫번째 DC 계수를 그대로 두고(델타 인코딩에서 시작점으로 사용), 그 이후 각 DC 계수에서 바로 이전 DC 계수를 뱨는 연산을 수행
    return torch.cat([dc, ac], dim=-1) # 델타 인코딩된 계수를 반환(데이터 중복성을 줄이고 압축률을 개선하는데 도움)

class DC_Predictor(nn.Module):# DC 값은 1개(0,0 point)
    def __init__(self):
        super(DC_Predictor, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.reshape(-1, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AC_Predictor(nn.Module):# AC 값은 63개 (0,0 제외) 8x8 block 기준
    def __init__(self):
        super(AC_Predictor, self).__init__()
        self.lstm = nn.LSTM(1, 16, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(32 * 63, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.reshape(-1, 63, 1)
        x, _ = self.lstm(x)
        x = x.reshape(-1, 32 * 63)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Model_bpp_estimator(nn.Module):#bpp 추정
    def __init__(self):
        super(Model_bpp_estimator, self).__init__()
        self.dc_predictor = DC_Predictor()
        self.ac_predictor = AC_Predictor()

    def forward(self, x):
        dc_cl = self.dc_predictor(x[..., 0])
        ac_cl = self.ac_predictor(x[..., 1:])
        outputs = dc_cl + ac_cl
        return outputs
    
@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
        bitEstimator = Model_bpp_estimator().to(device) #bpp estimator 인스턴스 생성
        bitEstimator.load_state_dict(torch.load('./bppmodel.pt')) # 가중치 가져오기(pretrained)
        # bitEstimator.half() if half else bitEstimator.float()
        # 모든 파라미터를 순회하며 freeze
        for param in bitEstimator.parameters():#bpp는 freeze 한다.
            param.requires_grad = False
    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        bitEstimator = Model_bpp_estimator().to(device) #bpp estimator 인스턴스 생성
        bitEstimator.load_state_dict(torch.load('./bppmodel.pt')) # 가중치 가져오기(pretrained)
        # bitEstimator.half() if half else bitEstimator.float()
        # 모든 파라미터를 순회하며 freeze
        for param in bitEstimator.parameters():#bpp는 freeze 한다.
            param.requires_grad = False
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    bitEstimator.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    loss = torch.zeros(3, device=device)
    bpp_loss_val = torch.zeros(1, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            # print("paths: ",paths)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            print("before interpolate im size:", im.size())
            x = im.size()
            # print(im.dtype)

            im_down = F.interpolate(im, scale_factor=1/2, mode='bilinear', align_corners=False)
            im_up = F.interpolate(im_down, size=(imgsz, imgsz), mode='bilinear', align_corners=False) # YOLOv5에 640x640 크기로 들어감
            im_down_2 = F.interpolate(im_up, scale_factor=1/2, mode='bilinear', align_corners=False)
            print("im size : ",im.size())

            #im_down_2= im

            resized_imgs_to_ycbcr = TJ_ycbcr.to_ycbcr(im_down_2, data_range = 1.0, half = False) # 0405코드 추가 // RGB 이미지를 YCbcr로 바꾸기(shape은 동일/ 단 픽셀은 [0,1] 값을 가짐)
            resized_imgs_to_ycbcr = torch.clamp(resized_imgs_to_ycbcr, 0, 1)
            #print("resized_imgs_to_ycbcr size : ",resized_imgs_to_ycbcr.size())
            quality = 85
            quantized_dct_y = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 0:1, :, :], quality, "luma") # 0405코드 추가 //Y채널(luma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            quantized_dct_cb = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 1:2, :, :], quality, "chroma") # 0405코드 추가 // Cb채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            quantized_dct_cr = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 2:3, :, :], quality, "chroma") # 0405코드 추가 // Cr채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            dequantized_dct_y = TJ_ijg.decompress_coefficients(quantized_dct_y , quality, "luma") # 0405코드 추가 //Y채널(luma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            dequantized_dct_cb = TJ_ijg.decompress_coefficients(quantized_dct_cb, quality, "chroma") # 0405코드 추가 // Cb채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            dequantized_dct_cr = TJ_ijg.decompress_coefficients(quantized_dct_cr, quality, "chroma") # 0405코드 추가 // Cr채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
            dequantized_dct = torch.cat([dequantized_dct_y, dequantized_dct_cb, dequantized_dct_cr], dim=1) #YCbCr 채널의 양자화된 DCT 계수를 하나의 텐서로 합치기(concat)
            quantized_dct = torch.cat([quantized_dct_y, quantized_dct_cb, quantized_dct_cr], dim=1) #YCbCr 채널의 양자화된 DCT 계수를 하나의 텐서로 합치기(concat)
            dequantized_dct = TJ_ycbcr.to_rgb(dequantized_dct, data_range = 1.0, half= False)
            # Ensure the values are in [0, 1] range before visualization and saving
            dequantized_dct = torch.clamp(dequantized_dct, 0, 1)
            #dequantized_dct = F.interpolate(dequantized_dct, size=(512, 512), mode='nearest', align_corners=False) # YOLOv5에 640x640 크기로 들어감
            # Visualize and save original and compressed images
           # visualize_and_save_images(im_jpeg[0], quantized_dct[0], 'Original Image', 'Compressed Image', f"batch_{batch_i}")
            #print("quantized_dct size : ",quantized_dct.size())
            blocks = TJ_block.blockify(quantized_dct, 8)
            blocks = rearrange(blocks, 'b c p h w -> b c p (h w)') # (B, C, 80 *80, 8, 8) -> (B, C, 80* 80, 64) // 여기서 p는 블록 갯수 (델타 인코딩을 하기 위해서)
            blocks = delta_encode(blocks) # 델타 인코딩 실시 :각 8x8블록의 첫번째 계수를 이용하여 연속된 블록간의 DC 계수 차이만 저장해서 데이터를 더욱 압축) (B, C, 80 * 80, 64) 데이터 압축
            blocks = rearrange(blocks, 'b c p (h w) -> b c p h w', h = 8, w = 8) #델타 인코딩 끝나면 다시 복원하기 (B,C, 80 * 80, 64) -> (B,C, 80*80, 8, 8)
            blocks = TJ_block.deblockify(blocks, (im_down_2.shape[2], im_down_2.shape[3])) 
            blocks = TJ_dct.zigzag(blocks) # (B,C, 640, 640) -> (B, C, L , 64) # zigzag 실시 안에서 8x8블록단위로 다시 처리하고 zigzag 순서에 따라 벡터화 하고있음.
            blocks = rearrange(blocks, "b c n co -> b (c n) co") # 여기서 n은 벡터화된 DCT 계수의 개수
            blocks = (torch.log(torch.abs(blocks) + 1)) / (torch.log(torch.Tensor([2]).to(device))) #로그 스케일 변환(데이터를 더 잘 처리할 수 있게)
            blocks = rearrange(blocks, "b cn co -> (b cn) co")

            scaled_imgs = F.interpolate(dequantized_dct, size=(x[2], x[3]), mode='bilinear', align_corners=False) 
            nb, _, height, width = scaled_imgs.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(scaled_imgs.float()) if compute_loss else (model(scaled_imgs.float(), augment=augment), None)
            #preds, train_out = model(dequantized_dct.float()) if compute_loss else (model(dequantized_dct.float(), augment=augment), None)
            # preds, train_out = model(quantized_dct.float()) if compute_loss else (model(quantized_dct.float(), augment=augment), None)
            pred_code_len=bitEstimator(blocks) # bpp 추정
            bpp_loss=rearrange(pred_code_len,'(b p1) 1  -> b p1',b=quantized_dct.shape[0])
            bpp_loss=torch.sum(bpp_loss, dim = 1)
            bpp_loss=torch.mean(bpp_loss) 
            bpp_loss=bpp_loss / (3*512*512)#(quantized_dct.shape[1]*quantized_dct.shape[2]*quantized_dct.shape[3])
            print('\nbpp_loss : ',bpp_loss)
            bpp_loss_val += bpp_loss
        

        # # Loss
        if compute_loss is not None:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls
            print('\nloss : ',loss)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f"val_batch{batch_i}_pred.jpg", names)  # pred

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    print("average bpp_loss : ",bpp_loss_val.cpu() / len(dataloader))
    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    """Parses command-line options for YOLOv5 model inference configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 tasks like training, validation, testing, speed, and study benchmarks based on provided
    options.
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
