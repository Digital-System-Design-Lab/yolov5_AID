![그림1](https://github.com/Digital-System-Design-Lab/yolov5_AID/assets/160388155/b3b147b2-0719-4e6e-95cf-c1503ab1dbb2)


train file 실행 예시

CUDA_VISIBLE_DEVICES=0 python train_downscaing.py --data VOC.yaml --imgsz 512 --hyp hyp.VOC.yaml  
--batch-size 32 --epochs 200  --weights VOC_epoch49_mAP_0.62045_imgsz_512_hyp_voc.pt --device 0 --project voc_test --name 1 --freeze 24

test file (inference) 실행 예시

CUDA_VISIBLE_DEVICES=4 python uniform_downscaling.py.py --data VOC.yaml --imgsz 512 --batch-size 4  --weights VOC_epoch49_mAP_0.62045_imgsz_512_hyp_voc.pt --device 4

