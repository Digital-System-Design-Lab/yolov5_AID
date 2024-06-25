# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val_downscaling as validate  # for end-of-epoch mAP (validation file name)
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    ExtendedModelEMA,
    ExtendedModelEMA2, #utils/torch_utils.py 
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import io
import torchvision.transforms as transforms
from torch.nn import InstanceNorm2d
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()

import torch.nn.init as init

#torchjpeg
import torchjpeg.torchjpeg.src.torchjpeg.dct as TJ_dct#torchjpeg 에서 dct 가져오기
import torchjpeg.torchjpeg.src.torchjpeg.quantization.ijg as TJ_ijg #torchjpeg에서 quantization 가져오기
import torchjpeg.torchjpeg.src.torchjpeg.dct._block as TJ_block 
import torchjpeg.torchjpeg.src.torchjpeg.dct._color as TJ_ycbcr 
from einops import rearrange # pip install einops (텐서를 좀 더 자유자재로 쓸수있게함)


torch.use_deterministic_algorithms(False, warn_only=True)

#weight 변화 확인용    
def print_model_weights(model):
    fc1 = model.fc1.weight.data
    fc2 = model.fc2.weight.data
    print(f"FC1 layer weight: {fc1}")
    print(f"\nFC2 layer weight: {fc2}")

#torchjpeg
def delta_encode(coefs):#DCT 계수에 대해 s/ 델타 인코딩 수행(데이터 압축에서 자주 사용되는 기법)
    #델타 인코딩은 연속된 데이터 사이의 차이(델타)만 저장하는 방식
    #입력은 DCT 계수 텐서(coefs) // 각 블록의 dc 계수에 대해 델타 인코딩 적용 / ac는 그대로 유지
    #coefs 크기 (B, C, H*W/64, 64)
    ac = coefs[..., 1:]             # b 1 4096 63 #나머지는 AC 계수
    dc = coefs[..., 0:1]            # b 1 4096 1 #첫번째 요소는 DC계수
    dc = torch.cat([dc[..., 0:1, :], dc[..., 1:, :] - dc[..., :-1, :]], dim=-2)#각 DC 계수에서 바로 이전 DC 계수를 빼는 연산
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


#gumbel softmax
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import time

"""
Gumbel Softmax Sampler
Requires 2D input [batchsize, number of categories]

Does not support sinlge binary category. Use two dimensions with softmax instead.
"""
verbose=False

class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False
        
    def cuda(self):
        self.gpu = True
    
    def cpu(self):
        self.gpu = False
        
    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        #noise = torch.cuda.FloatTensor(shape).uniform_()
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        end_time = time.time()
        uniform_samples_tensor = torch.cuda.FloatTensor(template_tensor.shape).uniform_()
        if verbose:
            print ('random', time.time() - end_time)
            end_time = time.time()

        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        if verbose:
            print( 'log', time.time() - end_time)
            end_time = time.time()
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = len(logits.shape) - 1
        end_time = time.time()

        gumble_samples_tensor = self.sample_gumbel_like(logits.data)

        if verbose:
            print ('gumble_sample', time.time() - end_time)
            end_time = time.time()

        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)

        if verbose:
            print ('gumble_trick_log_prob_samples', time.time() - end_time)
            end_time = time.time()

        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)

        if verbose:
            print ('soft_samples', time.time() - end_time)
            end_time = time.time()
        return soft_samples
    
    def gumbel_softmax(self, logits, temperature, hard=False, index=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [ ..., n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [..., n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """

        end_time = time.time()
        dim = len(logits.shape) - 1

        y = self.gumbel_softmax_sample(logits, temperature)

        if verbose:
            print ('gumbel_softmax_sample', time.time() - end_time)

        if hard:
            end_time = time.time()

            _, max_value_indexes = y.data.max(dim, keepdim=True)
#            y_hard = torch.zeros_like(logits).scatter_(1, max_value_indexes, 1)


            if verbose:
                print ('max_value_indexes', time.time() - end_time)
                end_time = time.time()

            y_hard = logits.data.clone().zero_().scatter_(dim, max_value_indexes, 1)


            if verbose:
                print ('y_hard', time.time() - end_time)
                end_time = time.time()

            y = Variable(y_hard - y.data) + y


            if verbose:
                print ('y', time.time() - end_time)
                end_time = time.time()
#            exit(1)

            if index:
                return idx
        return y
        
    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True) 


#downscaling model
class MyModelWithConditionalDownscaling(nn.Module):
    def __init__(self):
        super(MyModelWithConditionalDownscaling, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Identity()  # 마지막 fc 제외

        # ResNet pretrained 사용
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # fc 2 layer 추가
        self.fc1 = nn.Linear(512, 128)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

        self.gs = GumbleSoftmax()
        self.gs.cuda()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def downscale(self, x, factor):
        x = F.interpolate(x, scale_factor=1/factor, mode='bilinear', align_corners=False)
        #torch.use_deterministic_algorithms(False)
        x = F.interpolate(x, size=(512,512), mode='bilinear', align_corners=False)
        return x

    def forward(self, x, temperature=1):
        y = self.resnet18(x)
        y = torch.flatten(y, 1)
        y = F.relu(self.fc1(y))
        y = self.dropout(y)
        y = self.fc2(y)
        w = self.gs(y, temp=temperature, force_hard=True)

        out = 0
        df = 0
        factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        for i, factor in enumerate(factors):
            downscaled_img = self.downscale(x, factor) * w[:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            df += factor * w[:, i]
            out += downscaled_img

        print("\n","df", df)
        return out, df



def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")

    # Directories 
    w = save_dir / "weights"  # weights dir   runs/train/exp#/weights 
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt" #weights 폴더에 2개 추가

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")
            
        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset
      
    # Model
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak --weights

        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        gating_network=MyModelWithConditionalDownscaling().to(device)

        #torchjpeg
        bitEstimator = Model_bpp_estimator().to(device) #bpp estimator 인스턴스 생성
        bitEstimator.load_state_dict(torch.load('./bppmodel.pt')) # 가중치 가져오기(pretrained)
        # 모든 파라미터를 순회하며 freeze
        for param in bitEstimator.parameters():#bpp는 freeze 한다.
            param.requires_grad = False
            
        
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load


        if 'gating_network' in ckpt:  # ckpt(체크포인트) 안에 gating_network 상태가 있을 때만 로드
            gating_network.load_state_dict(ckpt['gating_network'])  # GatingNetwork 가중치 로드
            print("gating_network 가중치 pretrained 에서 가져옴.")
        else:
            LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
     


    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        gating_network=MyModelWithConditionalDownscaling().to(device)

        #torchjpeg
        bitEstimator = Model_bpp_estimator().to(device)
        bitEstimator.load_state_dict(torch.load('./bppmodel.pt')) # 가중치 가져오기(pretrained)
        # 모든 파라미터를 순회하며 freeze
        for param in bitEstimator.parameters():
            param.requires_grad = False

    amp = check_amp(model)  # check AMP
    

    
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    #stride : 이미지 크기 감소 비율을 결정
    #model.stride.max (모델이 한 번에 얼마나 많이 이미지를 다운샘플링 가는가?)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    #opt.imgsz : 사용자가 지정한 입력 이미지 크기 / gs : grid size, 
    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size 단일 GPU 환경에서 최적의 배치 크기 추정
        #RANK = -1 : 단일 gpu만 사용하는 경우
        #BATCH_SIZE = -1 : 사용자가 배치 크기를 명시적으로 지정 X인 경우
        batch_size = check_train_batch_size(model, imgsz, amp)#최적의 배치 크기 추정
        loggers.on_params_update({"batch_size": batch_size})#추정된 배치 크기를 로깅 시스템에 업데이트

    # Optimizer
    nbs = 64  # nominal batch size (이상적인 배치 크기)valex
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing // accumulate : 몇개의 배치를 처리한 후그래디언트를 업데이트 할지 결정하는 값
    #실제배치크기(batch_size)가 작을수록 더 많은 배치의 손실을 축적
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay (가중치 감소?) -> 과적합 방지를위해 손실함수에 추가하는 정규화
    #실제 배치 크기가 목표 배치크기(nbs)보다 작으면 가중치 감소를 증가시켜 보정
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    fn_optimizer = optim.Adam(gating_network.parameters(), lr=0.00005) ##******

    
    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    fn_scheduler = optim.lr_scheduler.StepLR(fn_optimizer, step_size=10, gamma=0.9)


    # EMA
    #ema = ModelEMA(model) if RANK in {-1, 0} else None
    ema = ExtendedModelEMA2(model, gating_network) if RANK in {-1, 0} else None


    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, fn_optimizer, ema, weights, epochs, resume) #edit yj
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model) 

    # SyncBatchNorm 필요없음
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader( #trainDataLoader
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,#변형
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
        drop_last=True
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader( #validation load val 데이터로더 (epoch 마다 1회 시행)
            val_path,
            imgsz,
            batch_size // WORLD_SIZE ,#batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision
            gating_network.half().float() #edit yj
        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches 배치수
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class #np.zeros:모든 요소가 0인 배열 생성 (초기화할때사용)
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class 로스 초기화
    callbacks.run("on_train_start")
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        
        callbacks.run("on_train_epoch_start") #utils/loggers/__init__.py
        model.train()
        gating_network.train()

        if opt.image_weights:#필요x
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx


        #print_model_weights(gating_network) # 가중치 출력 확인용

        mloss = torch.zeros(3, device=device)  # mean losses #torch.zeros : 모든 요소가 0인 텐서를 생성
        mtotal_loss = torch.zeros(1, device=device) #downscalingNetwork의 total loss를 기록하기 위함
        mbpp_loss = torch.zeros(1, device=device) #bpp loss 평균 기록
        mdf = torch.zeros(1, device=device)
        
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 10) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "total_loss","bpp_loss", "mdf", "Instances", "Size"))#total_loss, bpp_loss, mdf 추가함
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        fn_optimizer.zero_grad() #edit yj

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  

            # uint8 to float32, 0-255 to 0.0-1.0 텐서 데이터 타입을 float32로 변환(부동소수점 연산을 하기 위해)
            # 픽셀값을 0 ~ 1.0 범위로 정규화(성능과 학습 속도를 향상시키기 위해)
            #print('imgs size:',imgs.size()) #이거 하니까 (B,C,H,W) = (B,3,imgsz, imgsz)으로 고정되네
            # Warmup (학습률을 점진적으로 증가시키는 기법, 초기에 학습률이 너무 높아 발생하는 발산 방지)
            #즉, 초기에 낮은 학습률로 시작해서, 에포크동안 점차 원하는 학습률까지 증가
            if ni <= nw: #ni : 현재까지 처리한 배치 총 수 , nw: Warmup을 적용할 배치수(Warmup 기간동안에만 학습률 조절)
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())#np.interp: 선형 보간 함수
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale (필요없을듯..)
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp): #amp: 자동 혼합 정밀도 -> 그냥 계산 효율성, 성능 향상시키는 것 인듯..
                
                resized_imgs, df = gating_network(imgs)
                x = imgs.size()
                

                bpp_losses = []
                dequantized_preprocessed_imgs = []
                preprocessed_imgs = []

                for img, factor in zip(resized_imgs, df):
                    factor = factor.item()
                
                    # 이미지를 factor로 줄임
                    preprocessed_img = F.interpolate(img.unsqueeze(0), scale_factor=1/factor, mode='bilinear', align_corners=False)
                    preprocessed_imgs.append(preprocessed_img)  

                    #줄인 이미지를 bpp estimator로 넣음
                    resized_imgs_to_ycbcr = TJ_ycbcr.to_ycbcr(preprocessed_img, 1.0) # 0405코드 추가 // RGB 이미지를 YCbcr로 바꾸기(shape은 동일/ 단 픽셀은 [0,1] 값을 가짐)
 
                    quantized_dct_y = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 0:1, :, :], 85, "luma") # 0405코드 추가 //Y채널(luma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
                    quantized_dct_cb = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 1:2, :, :], 85, "chroma") # 0405코드 추가 // Cb채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
                    quantized_dct_cr = TJ_ijg.compress_coefficients(resized_imgs_to_ycbcr[:, 2:3, :, :], 85, "chroma") # 0405코드 추가 // Cr채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
                    dequantized_dct_y = TJ_ijg.decompress_coefficients(quantized_dct_y , 85, "luma") # 0405코드 추가 //Y채널(luma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
                    dequantized_dct_cb = TJ_ijg.decompress_coefficients(quantized_dct_cb, 85, "chroma") # 0405코드 추가 // Cb채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.
                    dequantized_dct_cr = TJ_ijg.decompress_coefficients(quantized_dct_cr, 85, "chroma") # 0405코드 추가 // Cr채널(chroma)에 대해 dct를 수행하고 나온 coefficient로 quality factor가 60인 경우에 맞게 양자화.


                    quantized_dct = torch.cat([quantized_dct_y, quantized_dct_cb, quantized_dct_cr], dim=1) #YCbCr 채널의 양자화된 DCT 계수를 하나의 텐서로 합치기(concat)
                    dequantized_dct = torch.cat([dequantized_dct_y, dequantized_dct_cb, dequantized_dct_cr], dim=1) #YCbCr 채널의 양자화된 DCT 계수를 하나의 텐서로 합치기(concat)
                    dequantized_dct = TJ_ycbcr.to_rgb(dequantized_dct, data_range = 1.0, half= False)
                    dequantized_dct = torch.clamp(dequantized_dct, 0, 1)
                    dequantized_preprocessed_imgs.append(dequantized_dct)
                    #dct 크기는 (B,C,H,W) (dct 계수들이 적혀잇음)

                    #compress_coefficients 안에서 dct 수행후 quality(20으로 조절)에 맞게 quantize 실시 
                    blocks = TJ_block.blockify(quantized_dct, 8) # 8의 의미: 8x8로 블록을 나누겠다는거임 , (B,C,H,W) -> (B,C,H/8 * W/8 ,8,8)  // (B,C,640, 640) -> (B, C, 80 * 80, 8, 8)
                    blocks = rearrange(blocks, 'b c p h w -> b c p (h w)') # (B, C, 80 *80, 8, 8) -> (B, C, 80* 80, 64) // 여기서 p는 블록 갯수
                    blocks = delta_encode(blocks) # 델타 인코딩 실시 (dc 값의 차이만 저장) (B, C, 80 * 80, 64) 데이터 압축
                    blocks = rearrange(blocks, 'b c p (h w) -> b c p h w', h = 8, w = 8) #(B,C, 80 * 80, 64) -> (B,C, 80*80, 8, 8)
                   
                    #blocks = TJ_block.deblockify(blocks, (imgsz, imgsz)) #(B,C, 80*80, 8, 8) -> (B, C, 640, 640) 블록으로 나눠져 있던것을 다시 복원(블록 합치기?)
                    blocks = TJ_block.deblockify(blocks, (preprocessed_img.shape[2], preprocessed_img.shape[3]))
                    
                    blocks = TJ_dct.zigzag(blocks) # (B,C, 640, 640) -> (B, C, L , 64) # zigzag 실시
                    blocks = rearrange(blocks, "b c n co -> b (c n) co") # 여기서 n은 벡터화된 DCT 계수의 개수
                    blocks = (torch.log(torch.abs(blocks) + 1)) / (torch.log(torch.Tensor([2]).to(device)))
                    blocks = rearrange(blocks, "b cn co -> (b cn) co")
                    pred_code_len=bitEstimator(blocks) # bpp 추정
                    bpp_loss=rearrange(pred_code_len,'(b p1) 1  -> b p1',b=preprocessed_img.shape[0])
                    bpp_loss=torch.sum(bpp_loss, dim = 1)
                    bpp_loss=torch.mean(bpp_loss) 
                    bpp_loss=bpp_loss / (imgs.shape[1]*imgs.shape[2]*imgs.shape[3])

                    bpp_losses.append(bpp_loss)
                    # print('\nbpp_loss : ',bpp_loss)


                # 배치의 최종 bpp_loss 계산
                bpp_loss = torch.mean(torch.stack(bpp_losses))

                # 각 이미지를 다시 원본 크기로 조정하여 모델에 넣기
                torch.use_deterministic_algorithms(False)
                scaled_imgs = [F.interpolate(img, size=(x[2], x[3]), mode='bilinear', align_corners=False) for img in dequantized_preprocessed_imgs]
                scaled_imgs = torch.cat(scaled_imgs, dim=0)

                pred = model(scaled_imgs) #syh_edit 줄였다 늘렸다 한것을 YOLOv5에 넣음
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size // pred결과와 target 결과를 기반으로 loss 계산(yolov5)


                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    #WORLD_SIZE: 전체 gpu수
                if opt.quad:#quad가 활성화 되어있으면 손실을 4배로 증가
                    loss *= 4.0

                total_loss = bpp_loss * 2 + loss * 1


                print('yolov5 loss : ',loss)
                print('bpp_loss: ',bpp_loss)
                print('total_loss : ',total_loss)
            # Backward
            scaler.scale(total_loss).backward()#yolov5 모델의 파라미터 업데이트 0302: 하나만 backward


            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                scaler.unscale_(fn_optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                torch.nn.utils.clip_grad_norm_(gating_network.parameters(), max_norm=10.0)


                scaler.step(fn_optimizer)
                scaler.step(optimizer)  # optimizer.step
                scaler.update()

                optimizer.zero_grad()
                fn_optimizer.zero_grad()

                if ema:
                    ema.update(model, gating_network)

                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mtotal_loss = (mtotal_loss * i + total_loss) / (i+ 1) # total_loss 평균 업데이트
                mbpp_loss = (mbpp_loss * i + bpp_loss) / (i + 1) # update bpp_loss 평균 업데이트
                mdf = (mdf * i + df) / (i + 1) # update df(downscaling factor) 평균 업데이트
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 8)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, mtotal_loss.item(), mbpp_loss.item(), mdf.mean().item(), targets.shape[0], imgs.shape[-1])#mbpp, mtotal_loss 추가
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        fn_scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)

            ema.update_attr(model, gating_network, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])


            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run( #검증 데이터셋에 대해 모델 성능 평가 (epoch 1회 끝나고 시행)
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE ,
                    imgsz=imgsz,
                    half=amp,
                    model = ema.model_ema,
                    gating_network = ema.gating_ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss
                )

            # Update best mAP // mAP가 가장 높은것을 best.pt로 만듦.
            fi = fitness(np.array(results[:-2]).reshape(1, -1)) 
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:#현재 epoch의 mAP값이 이전에 기록된 최고 mAP보다 높으면 현재 모델의 상태를 best.pt로 저장
                best_fitness = fi

            mdf_mean = np.mean([df.mean().item() for df in mdf])

            log_vals = list(mloss) + [mtotal_loss.item()] +[mbpp_loss.item()] +[mdf_mean]+ list(results) + lr #result.csv에 저장 // mtotal_loss, mbpp_loss 추가
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)
            # 에폭 종료 후 가중치 로깅
            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = { #checkpoint 인듯
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.model_ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "fn_optimizer" : fn_optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                    "gating_network": deepcopy(gating_network.state_dict()),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:# f -> best.pt의 경로
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run( #train이 끝나고 나서도 validation 시행
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE ,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        gating_network_weights = f #edit yj
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results)  + [mtotal_loss.item()] + [mbpp_loss.item()] + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results) 

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mixup (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),
        }  # segment copy-paste (probability)

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        initial_values = []

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = ( #result.png에 적히는 key들?
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


def generate_individual(input_ranges, individual_length):
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
