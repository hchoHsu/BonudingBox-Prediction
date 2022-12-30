# Bounding Box Module執行
:::info
檔案位置：```/home/elsalab/Desktop/23/husky_ws/src/bbox/src/scripts/models/```
:::


[Model下載連結](https://drive.google.com/file/d/1yH0htWdx-N8ahsM7bREstx4QgPunzv5j/view?usp=share_link)

## Pytorch轉TensorRT

Log 細節可參考：[TensorRT-Log](./Logs/TensorRT-Log.md)

```bash
python3 tools/trt.py -f ./yolox/exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
```

改fp16可以改tools/trt.py line 57
```bash=54
model_trt = torch2trt(
    model,
    [x],
    fp16_mode=True,
    log_level=trt.Logger.INFO,
    max_workspace_size=(1 << 30),
)
```

Output的models會在```YOLOX_outputs/yolox_s_mix_det/```

TensorRT轉fp16會有subnormal fp16 values detected，不確定影響。

## Inference測試
```bash
python3 tools/inference.py
```

目前因為torchvision版本問題，inference應該無法跑fp16以外的models
參考：https://blog.csdn.net/qq_42257666/article/details/125293727

要改成TensorRT inference請將tools/inference.py line 285改為True
```bash=278
args.name = None
args.fp16 = True
args.device = "cuda"
args.ablation = False
args.mot20 = False
args.fps = 30
args.batch_size = 1
args.trt = True
```

![](https://i.imgur.com/SjCjso6.png)
