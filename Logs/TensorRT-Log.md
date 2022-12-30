# BoT-SORT to TensorRT

## Alice Error Log:

+ Command:
```bash
$ python3 tools/trt.py -f \
          yolox/exps/example/mot/yolox_s_mix_det.py -c \
          pretrained/bytetrack_s_mot17.pth.tar
```
+ Error msg:
![](https://i.imgur.com/8QIfMbO.png)

+ Current Version of Torch, TensorRT, cuDNN
    + torch.__version__: '1.12.1'
    + tensorrt.__version__: '7.2.3.4'
    + torch2trt (conda list): '0.4.0'
    + cuDNN: '8.0.5'

+ Command Used to solve the problem:
    + 2022/12/28:
        + The command failed, same error occured on alice.
    1. Upgrade tensorrt
    ```bash
    $ python3 -m pip install --upgrade tensorrt
    ```
    2. Change the max_workspace_size from 1<<32 to 1<<30 in ```tools/trt.py```
    3. Make the correct directory
    ```bash
    $ mkdir deploy
    $ mkdir deploy/TensorRT
    $ mkdir deploy/TensorRT/cpp
    ```

## TensorRT (The command of Alice and Orin is the same.)

### FP16 = False

+ TensorRT Log
```bash
(venv) elsalab@ubuntu:~/.../models$ python3 tools/trt.py -f ./yolox/exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension:
  warn(f"Failed to load image Python extension: {e}")
2022-12-30 14:03:06.728 | INFO     | __main__:main:51 - loaded checkpoint done.
[12/30/2022-14:03:10] [TRT] [I] [MemUsageChange] Init CUDA: CPU +213, GPU +0, now: CPU 2103, GPU 12148 (MiB)
[12/30/2022-14:03:13] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +351, GPU +461, now: CPU 2473, GPU 12623 (MiB)
[12/30/2022-14:03:15] [TRT] [E] 4: [layers.cpp::estimateOutputDims::1954] Error Code 4: Internal Error ((Unnamed Layer* 1500) [Concatenation]: all concat input tensors must have the same dimensions except on the concatenation axis (0), but dimensions mismatched at index 2. Input 0 shape: [1,6,10336], Input 1 shape: [1,6,2584])
[12/30/2022-14:03:15] [TRT] [E] 4: [layers.cpp::estimateOutputDims::1954] Error Code 4: Internal Error ((Unnamed Layer* 1500) [Concatenation]: all concat input tensors must have the same dimensions except on the concatenation axis (0), but dimensions mismatched at index 2. Input 0 shape: [1,6,10336], Input 1 shape: [1,6,2584])
[12/30/2022-14:03:15] [TRT] [W] Tensor DataType is determined at build time for tensors not marked as input or output.
[12/30/2022-14:03:16] [TRT] [I] ---------- Layers Running on DLA ----------
[12/30/2022-14:03:16] [TRT] [I] ---------- Layers Running on GPU ----------
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: Reformatting CopyNode for Network Input input_0
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:0:SLICE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:1:SLICE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:2:SLICE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:3:SLICE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 149) [Slice]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.stem.conv.conv:0:CONVOLUTION:GPU + backbone.backbone.stem.conv.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.stem.conv.act:0:SIGMOID:GPU), backbone.backbone.stem.conv.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark2.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark2.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark2.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark2.1.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.m.0.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark2.1.m.0:0:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark3.1.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark3.1.m.0:0:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark3.1.m.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark3.1.m.1:0:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark3.1.m.2.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark3.1.m.2:0:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark4.1.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark4.1.m.0:0:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark4.1.m.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark4.1.m.1:0:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark4.1.m.2.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark4.1.m.2:0:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POOLING: backbone.backbone.dark5.1.m.0:0:MAX:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POOLING: backbone.backbone.dark5.1.m.1:0:MAX:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POOLING: backbone.backbone.dark5.1.m.2:0:MAX:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 560) [ElementWise]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.2.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] RESIZE: backbone.upsample:0:RESIZE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 732) [Resize]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p4.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p4.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] RESIZE: backbone.upsample:1:RESIZE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 851) [Resize]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p3.conv1.act:0:SIGMOID:GPU), backbone.C3_p3.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p3.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 850) [ElementWise]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.0.0.act:0:SIGMOID:GPU), head.cls_convs.0.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.0.0.act:0:SIGMOID:GPU), head.reg_convs.0.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n3.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n3.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_preds.0:0:CONVOLUTION:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:1:SIGMOID:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:0:SIGMOID:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1223) [Convolution]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1225) [Activation]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SHUFFLE: head:14:SHUFFLE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: head:14:SHUFFLE:GPU_copy_output
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 731) [ElementWise]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.1.0.act:0:SIGMOID:GPU), head.cls_convs.1.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.1.0.act:0:SIGMOID:GPU), head.reg_convs.1.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n4.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n4.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_preds.1:0:CONVOLUTION:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:4:SIGMOID:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:3:SIGMOID:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1318) [Convolution]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1320) [Activation]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SHUFFLE: head:20:SHUFFLE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: head:20:SHUFFLE:GPU_copy_output
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.2.0.act:0:SIGMOID:GPU), head.cls_convs.2.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.2.0.act:0:SIGMOID:GPU), head.reg_convs.2.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_preds.2:0:CONVOLUTION:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:7:SIGMOID:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:6:SIGMOID:GPU)
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1413) [Convolution]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1415) [Activation]_output copy
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SHUFFLE: head:26:SHUFFLE:GPU
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] COPY: head:26:SHUFFLE:GPU_copy_output
[12/30/2022-14:03:16] [TRT] [I] [GpuLayer] SHUFFLE: head:28:SHUFFLE:GPU
[12/30/2022-14:03:16] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2547, GPU 13287 (MiB)
[12/30/2022-14:03:16] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 2547, GPU 13295 (MiB)
[12/30/2022-14:03:16] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[12/30/2022-14:04:47] [TRT] [I] Some tactics do not have sufficient workspace memory to run. Increasing workspace size will enable more tactics, please check verbose output for requested sizes.
[12/30/2022-14:05:50] [TRT] [I] Detected 1 inputs and 1 output network tensors.
[12/30/2022-14:05:50] [TRT] [I] Total Host Persistent Memory: 211680
[12/30/2022-14:05:50] [TRT] [I] Total Device Persistent Memory: 0
[12/30/2022-14:05:50] [TRT] [I] Total Scratch Memory: 0
[12/30/2022-14:05:50] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 40 MiB, GPU 564 MiB
[12/30/2022-14:05:50] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 72.2336ms to assign 9 blocks to 219 nodes requiring 57876484 bytes.
[12/30/2022-14:05:50] [TRT] [I] Total Activation Memory: 57876484
[12/30/2022-14:05:50] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +1, GPU +6, now: CPU 2566, GPU 13409 (MiB)
[12/30/2022-14:05:50] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +34, GPU +64, now: CPU 34, GPU 64 (MiB)
[12/30/2022-14:05:50] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +5, now: CPU 2565, GPU 13394 (MiB)
[12/30/2022-14:05:50] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +55, now: CPU 34, GPU 119 (MiB)
[12/30/2022-14:05:50] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[12/30/2022-14:05:50] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
2022-12-30 14:05:50.881 | INFO     | __main__:main:64 - Converted TensorRT model done.
[12/30/2022-14:05:50] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[12/30/2022-14:05:50] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
2022-12-30 14:05:51.277 | INFO     | __main__:main:72 - Converted TensorRT model engine file is saved for C++ inference.
```

+ Inference的Log與輸出
```bash
(venv) elsalab@ubuntu:~/.../models$ python3 tools/inference.py
/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension:
  warn(f"Failed to load image Python extension: {e}")
/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torch/functional.py:478: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2894.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Traceback (most recent call last):
  File "tools/inference.py", line 332, in <module>
    test_inference(predictor, tracker, args, exp)
  File "tools/inference.py", line 269, in test_inference
    img = inference(predictor, tracker, args, exp, img_path)
  File "tools/inference.py", line 194, in inference
    outputs, img_info = predictor.inference(img)
  File "tools/inference.py", line 188, in inference
    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
  File "/home/elsalab/Desktop/23/husky_ws/src/bbox_model/src/scripts/models/./yolox/utils/boxes.py", line 57, in postprocess
    nms_out_index = torchvision.ops.batched_nms(
  File "/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/ops/boxes.py", line 75, in batched_nms
    return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)
  File "/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torch/jit/_trace.py", line 1127, in wrapper
    return fn(*args, **kwargs)
  File "/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/ops/boxes.py", line 94, in _batched_nms_coordinate_trick
    keep = nms(boxes_for_nms, scores, iou_threshold)
  File "/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/ops/boxes.py", line 40, in nms
    _assert_has_ops()
  File "/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/extension.py", line 48, in _assert_has_ops
    raise RuntimeError(
RuntimeError: Couldn't load custom C++ ops. This can happen if your PyTorch and torchvision versions are incompatible, or if you had errors while compiling torchvision from source. For further information on the compatible versions, check https://github.com/pytorch/vision#installation for the compatibility matrix. Please check your PyTorch version with torch.__version__ and your torchvision version with torchvision.__version__ and verify if they are compatible, and if not please reinstall torchvision so that it matches your PyTorch install.
```

### FP16 = True

+ TensorRT Log
輸出訊息Log超過Hackmd篇幅限制("Subnormal FP16 values detected." ANYWHERE!)，請至此連結處看訊息輸出內容：[fp16=True](./fp16%3DTrue.md)
```bash
(venv) ~/.../models$ python3 tools/trt.py -f ./yolox/exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
...
```

+ Inference Log和輸出
```bash
(venv) elsalab@ubuntu:~/.../models$ python tools/inference.py
/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension:
  warn(f"Failed to load image Python extension: {e}")
/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torch/functional.py:478: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2894.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
../../NTHU_23/20220917_111613534391/image_raw/2.png (376, 672, 3)
../../NTHU_23/20220917_111613534391/image_raw/3.png (376, 672, 3)
../../NTHU_23/20220917_111613534391/image_raw/4.png (376, 672, 3)
../../NTHU_23/20220917_111613534391/image_raw/5.png (376, 672, 3)
.
. (略)
.
../../NTHU_23/20220917_111613534391/image_raw/719.png (376, 672, 3)
../../NTHU_23/20220917_111613534391/image_raw/720.png (376, 672, 3)
../../NTHU_23/20220917_111613534391/image_raw/721.png (376, 672, 3)
../../NTHU_23/20220917_111613534391/image_raw/722.png (376, 672, 3)
frame num:721, fps:12.629098206103572
```

+ Input Image，Orin Output Image 和 Alice Output Image比較：
![原圖](https://i.imgur.com/mDoLkBS.png)
![fp16](https://i.imgur.com/R75XuhR.png)
![alice](https://i.imgur.com/WsXY3lI.png)
