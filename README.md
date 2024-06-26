# Adaptive Image Downscaling

- **Project Name**: Adaptive Image Downscaling
- **Project Member**: 김예지, 서영홍

## ❤️ 요약

본 연구는 에지 디바이스에서 서버로 이미지를 그대로 보내지 않고, JPEG과 같은 코덱을 이용하여 이미지를 압축 후 서버로 전송함으로써 전송 데이터의 크기를 줄인다. 
이미지 압축 및 전송 이전에 이미지 마다의 적합한 Factor로  Downscaling 하여 JPEG 압축 효율성을 증가시키고 데이터 전송 효율(Bitrate)과 Machine Vision Task의 정확도(Accuracy) Optimization 한다


## ❤️ 전체 architecture

![그림2](https://github.com/Digital-System-Design-Lab/yolov5_AID/assets/160388155/442948ba-5af5-4685-a17d-79233842e13b)


## ❤️ 성능 평가
![그림1](https://github.com/Digital-System-Design-Lab/yolov5_AID/assets/160388155/b3b147b2-0719-4e6e-95cf-c1503ab1dbb2)


train file 실행 예시
```bash
CUDA_VISIBLE_DEVICES=0 python train_downscaing.py --data VOC.yaml --imgsz 512 --hyp hyp.VOC.yaml --batch-size 32 --epochs 200 --weights VOC_epoch49_mAP_0.62045_imgsz_512_hyp_voc.pt --device 0 --project voc_test --name 1 --freeze 24
```

test file (inference) 실행 예시
```bash
CUDA_VISIBLE_DEVICES=4 python uniform_downscaling.py.py --data VOC.yaml --imgsz 512 --batch-size 4 --weights VOC_epoch49_mAP_0.62045_imgsz_512_hyp_voc.pt --device 4
```

