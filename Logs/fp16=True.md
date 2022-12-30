> This note is the log of tensorRT with fp16=True

```bash
/home/elsalab/Desktop/23/husky_ws/venv/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension:
  warn(f"Failed to load image Python extension: {e}")
2022-12-30 14:14:34.126 | INFO     | __main__:main:51 - loaded checkpoint done.
[12/30/2022-14:14:38] [TRT] [I] [MemUsageChange] Init CUDA: CPU +213, GPU +0, now: CPU 2103, GPU 12191 (MiB)
[12/30/2022-14:14:41] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +351, GPU +446, now: CPU 2473, GPU 12655 (MiB)
[12/30/2022-14:14:43] [TRT] [E] 4: [layers.cpp::estimateOutputDims::1954] Error Code 4: Internal Error ((Unnamed Layer* 1500) [Concatenation]: all concat input tensors must have the same dimensions except on the concatenation axis (0), but dimensions mismatched at index 2. Input 0 shape: [1,6,10336], Input 1 shape: [1,6,2584])
[12/30/2022-14:14:43] [TRT] [E] 4: [layers.cpp::estimateOutputDims::1954] Error Code 4: Internal Error ((Unnamed Layer* 1500) [Concatenation]: all concat input tensors must have the same dimensions except on the concatenation axis (0), but dimensions mismatched at index 2. Input 0 shape: [1,6,10336], Input 1 shape: [1,6,2584])
[12/30/2022-14:14:43] [TRT] [W] Tensor DataType is determined at build time for tensors not marked as input or output.
[12/30/2022-14:14:43] [TRT] [I] ---------- Layers Running on DLA ----------
[12/30/2022-14:14:43] [TRT] [I] ---------- Layers Running on GPU ----------
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: Reformatting CopyNode for Network Input input_0
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:0:SLICE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:1:SLICE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:2:SLICE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SLICE: backbone.backbone.stem:3:SLICE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 149) [Slice]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.stem.conv.conv:0:CONVOLUTION:GPU + backbone.backbone.stem.conv.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.stem.conv.act:0:SIGMOID:GPU), backbone.backbone.stem.conv.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark2.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark2.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark2.1.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.m.0.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark2.1.m.0:0:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark3.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark3.1.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark3.1.m.0:0:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark3.1.m.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark3.1.m.1:0:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark3.1.m.2.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark3.1.m.2:0:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark4.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark4.1.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark4.1.m.0:0:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark4.1.m.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark4.1.m.1:0:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(PWN(backbone.backbone.dark4.1.m.2.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv2.act:1:ELEMENTWISE:GPU), backbone.backbone.dark4.1.m.2:0:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POOLING: backbone.backbone.dark5.1.m.0:0:MAX:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POOLING: backbone.backbone.dark5.1.m.1:0:MAX:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POOLING: backbone.backbone.dark5.1.m.2:0:MAX:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 560) [ElementWise]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.backbone.dark5.2.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] RESIZE: backbone.upsample:0:RESIZE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 732) [Resize]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p4.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p4.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] RESIZE: backbone.upsample:1:RESIZE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 851) [Resize]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p3.conv1.act:0:SIGMOID:GPU), backbone.C3_p3.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_p3.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 850) [ElementWise]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.0.0.act:0:SIGMOID:GPU), head.cls_convs.0.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.0.0.act:0:SIGMOID:GPU), head.reg_convs.0.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n3.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n3.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:0:SIGMOID:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1223) [Convolution]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1225) [Activation]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SHUFFLE: head:14:SHUFFLE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: head:14:SHUFFLE:GPU_copy_output
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 731) [ElementWise]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.1.0.act:0:SIGMOID:GPU), head.cls_convs.1.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.1.0.act:0:SIGMOID:GPU), head.reg_convs.1.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n4.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(backbone.C3_n4.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_preds.1:0:CONVOLUTION:GPU + PWN(head:4:SIGMOID:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:3:SIGMOID:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1318) [Convolution]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1320) [Activation]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SHUFFLE: head:20:SHUFFLE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: head:20:SHUFFLE:GPU_copy_output
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.cls_convs.2.0.act:0:SIGMOID:GPU), head.cls_convs.2.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(PWN(head.reg_convs.2.0.act:0:SIGMOID:GPU), head.reg_convs.2.0.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.cls_preds.2:0:CONVOLUTION:GPU + PWN(head:7:SIGMOID:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] CONVOLUTION: head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] POINTWISE: PWN(head:6:SIGMOID:GPU)
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1413) [Convolution]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: (Unnamed Layer* 1415) [Activation]_output copy
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SHUFFLE: head:26:SHUFFLE:GPU
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] COPY: head:26:SHUFFLE:GPU_copy_output
[12/30/2022-14:14:43] [TRT] [I] [GpuLayer] SHUFFLE: head:28:SHUFFLE:GPU
[12/30/2022-14:14:43] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +1, GPU +8, now: CPU 2547, GPU 13318 (MiB)
[12/30/2022-14:14:43] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 2547, GPU 13326 (MiB)
[12/30/2022-14:14:43] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[12/30/2022-14:15:46] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:47] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:47] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:47] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:47] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:47] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:47] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:47] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:47] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:47] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:47] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:47] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:47] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:47] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:47] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:47] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:47] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:47] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:47] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:48] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:15:48] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:48] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:50] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:51] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:51] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:51] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:15:51] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:15:51] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:15:51] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:01] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:01] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:01] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:01] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:01] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:01] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:01] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:01] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:01] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:01] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:01] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:01] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:01] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:01] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:01] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:01] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:01] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:01] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:01] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:01] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:01] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:17:02] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:17:02] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:17:02] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:09] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:10] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:10] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:15] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:15] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:15] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:15] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:15] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:15] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:15] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:16] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:16] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:16] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:18:16] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:17] [TRT] [I] Some tactics do not have sufficient workspace memory to run. Increasing workspace size will enable more tactics, please check verbose output for requested sizes.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:18] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:24] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:25] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:25] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:18:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:18:27] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:18:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:18:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:11] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:12] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:12] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:12] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:12] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:12] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:22] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:23] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:23] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:26] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:26] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:26] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:26] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:26] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:26] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:26] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:28] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:28] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:28] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:30] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:36] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:37] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:37] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:38] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:38] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:38] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:38] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:38] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:38] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:38] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:39] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:19:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:45] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:45] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:45] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:46] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:49] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:49] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:49] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:49] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:50] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:50] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:50] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:50] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:50] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:50] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:50] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:50] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:50] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:50] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:50] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:50] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:50] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:50] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:51] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:51] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:51] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:19:51] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:19:56] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:19:56] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:19:56] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:03] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:03] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:04] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:04] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:04] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:04] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:06] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:06] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:06] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:11] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:14] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:14] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:15] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:15] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:15] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:15] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:15] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:15] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:15] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:15] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:15] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:15] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:20] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:20] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:20] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:20] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:21] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:21] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:21] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:21] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:21] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:21] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:21] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:21] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:22] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:23] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:24] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:26] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:26] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:26] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:26] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:26] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:26] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:26] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:26] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:26] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:26] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:26] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:26] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:26] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:26] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:26] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:26] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:27] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:27] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:20:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:32] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:32] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:32] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:33] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:33] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:33] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:33] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:33] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:33] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:37] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:37] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:37] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:38] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:38] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:38] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:38] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:42] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:42] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:42] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:42] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:20:42] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:42] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:45] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:45] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:45] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:45] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:45] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:45] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:45] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:45] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:45] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:45] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:45] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:45] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:46] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:46] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:46] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:46] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:46] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:47] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:47] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:47] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:54] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:54] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:54] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:55] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:55] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:55] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:55] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:55] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:55] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:58] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:58] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:58] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:58] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:58] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:58] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:58] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:58] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:58] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:58] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:58] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:58] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:58] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:58] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:20:59] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:20:59] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:20:59] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:00] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:00] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:00] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:02] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:02] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:02] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:03] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:03] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:03] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:03] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:03] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:03] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:03] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:03] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:04] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:04] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:10] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:10] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:10] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:10] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:10] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:10] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:10] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:11] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:12] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:12] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:12] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:17] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:18] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:22] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:22] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:24] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:21:25] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:21:25] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:21:25] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:07] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:07] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:07] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:08] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:08] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:08] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:08] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:09] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:09] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:09] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:09] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:09] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:09] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:09] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:09] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:09] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:09] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:09] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:09] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:09] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:09] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:10] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:10] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:10] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:10] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:11] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:11] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:11] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:13] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:13] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:13] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:13] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:13] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:13] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:13] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:13] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:13] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:13] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:13] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:13] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:13] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:13] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:13] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:13] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:13] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:13] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:13] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:13] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:13] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:13] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:13] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:13] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:13] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:13] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:13] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:13] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:14] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:14] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:14] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:17] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:17] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:17] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:17] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:17] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:17] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:17] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:17] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:17] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:18] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:18] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:18] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:19] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:19] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:19] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:22] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:22] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:22] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:23] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:23] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:23] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:23] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:24] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:24] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:24] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:27] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:27] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:27] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:28] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:28] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:28] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:28] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:28] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:28] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:29] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:29] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:29] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:29] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:29] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:29] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:29] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:29] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:29] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:29] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:29] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:29] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:29] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:29] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:29] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:29] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:29] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:29] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:29] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:29] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:29] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:30] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:30] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:30] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:33] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:33] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:33] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:34] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:34] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:34] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:34] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:34] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:34] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:34] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:34] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:34] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:34] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:34] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:34] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:34] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:34] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:34] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:34] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:34] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:34] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:35] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:35] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:35] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:36] [TRT] [W] Weights [name=head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:36] [TRT] [W] Weights [name=head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:36] [TRT] [W] Weights [name=head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:36] [TRT] [W] Weights [name=head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:36] [TRT] [W] Weights [name=head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:36] [TRT] [W] Weights [name=head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:36] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:36] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:39] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:39] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:39] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [I] Detected 1 inputs and 1 output network tensors.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark2.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.0.act:0:SIGMOID:GPU), backbone.backbone.dark2.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark2.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark2.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark2.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark2.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark2.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark2.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.0.act:0:SIGMOID:GPU), backbone.backbone.dark3.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark3.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark3.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark3.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark3.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark3.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.0.act:0:SIGMOID:GPU), backbone.backbone.dark4.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv1.bn:0:SCALE:GPU || backbone.backbone.dark4.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.0.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.1.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.m.2.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.m.2.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.m.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.m.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark4.1.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark4.1.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark4.1.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark4.1.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark5.0.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.0.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.0.act:0:SIGMOID:GPU), backbone.backbone.dark5.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark5.1.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.1.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.1.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.1.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv1.bn:0:SCALE:GPU || backbone.backbone.dark5.2.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv1.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark5.2.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.m.0.conv2.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.backbone.dark5.2.conv3.conv:0:CONVOLUTION:GPU + backbone.backbone.dark5.2.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.backbone.dark5.2.conv3.act:0:SIGMOID:GPU), backbone.backbone.dark5.2.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.lateral_conv0.conv:0:CONVOLUTION:GPU + backbone.lateral_conv0.bn:0:SCALE:GPU + PWN(PWN(backbone.lateral_conv0.act:0:SIGMOID:GPU), backbone.lateral_conv0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_p4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv1.bn:0:SCALE:GPU || backbone.C3_p4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_p4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_p4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p4.conv3.act:0:SIGMOID:GPU), backbone.C3_p4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.reduce_conv1.conv:0:CONVOLUTION:GPU + backbone.reduce_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.reduce_conv1.act:0:SIGMOID:GPU), backbone.reduce_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_p3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv1.bn:0:SCALE:GPU || backbone.C3_p3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_p3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_p3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_p3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_p3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_p3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_p3.conv3.act:0:SIGMOID:GPU), backbone.C3_p3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.bu_conv2.conv:0:CONVOLUTION:GPU + backbone.bu_conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv2.act:0:SIGMOID:GPU), backbone.bu_conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.stems.0.conv:0:CONVOLUTION:GPU + head.stems.0.bn:0:SCALE:GPU + PWN(PWN(head.stems.0.act:0:SIGMOID:GPU), head.stems.0.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.cls_convs.0.0.conv:0:CONVOLUTION:GPU + head.cls_convs.0.0.bn:0:SCALE:GPU || head.reg_convs.0.0.conv:0:CONVOLUTION:GPU + head.reg_convs.0.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n3.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv1.bn:0:SCALE:GPU || backbone.C3_n3.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.cls_convs.0.1.conv:0:CONVOLUTION:GPU + head.cls_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.0.1.act:0:SIGMOID:GPU), head.cls_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.reg_convs.0.1.conv:0:CONVOLUTION:GPU + head.reg_convs.0.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.0.1.act:0:SIGMOID:GPU), head.reg_convs.0.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.cls_preds.0:0:CONVOLUTION:GPU + PWN(head:1:SIGMOID:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.reg_preds.0:0:CONVOLUTION:GPU || head.obj_preds.0:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n3.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n3.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n3.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n3.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n3.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n3.conv3.act:0:SIGMOID:GPU), backbone.C3_n3.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.bu_conv1.conv:0:CONVOLUTION:GPU + backbone.bu_conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.bu_conv1.act:0:SIGMOID:GPU), backbone.bu_conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.stems.1.conv:0:CONVOLUTION:GPU + head.stems.1.bn:0:SCALE:GPU + PWN(PWN(head.stems.1.act:0:SIGMOID:GPU), head.stems.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.cls_convs.1.0.conv:0:CONVOLUTION:GPU + head.cls_convs.1.0.bn:0:SCALE:GPU || head.reg_convs.1.0.conv:0:CONVOLUTION:GPU + head.reg_convs.1.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n4.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv1.bn:0:SCALE:GPU || backbone.C3_n4.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv2.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.cls_convs.1.1.conv:0:CONVOLUTION:GPU + head.cls_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.1.1.act:0:SIGMOID:GPU), head.cls_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.reg_convs.1.1.conv:0:CONVOLUTION:GPU + head.reg_convs.1.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.1.1.act:0:SIGMOID:GPU), head.reg_convs.1.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv1.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv1.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv1.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.reg_preds.1:0:CONVOLUTION:GPU || head.obj_preds.1:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n4.m.0.conv2.conv:0:CONVOLUTION:GPU + backbone.C3_n4.m.0.conv2.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.m.0.conv2.act:0:SIGMOID:GPU), backbone.C3_n4.m.0.conv2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=backbone.C3_n4.conv3.conv:0:CONVOLUTION:GPU + backbone.C3_n4.conv3.bn:0:SCALE:GPU + PWN(PWN(backbone.C3_n4.conv3.act:0:SIGMOID:GPU), backbone.C3_n4.conv3.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.stems.2.conv:0:CONVOLUTION:GPU + head.stems.2.bn:0:SCALE:GPU + PWN(PWN(head.stems.2.act:0:SIGMOID:GPU), head.stems.2.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.cls_convs.2.0.conv:0:CONVOLUTION:GPU + head.cls_convs.2.0.bn:0:SCALE:GPU || head.reg_convs.2.0.conv:0:CONVOLUTION:GPU + head.reg_convs.2.0.bn:0:SCALE:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.cls_convs.2.1.conv:0:CONVOLUTION:GPU + head.cls_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.cls_convs.2.1.act:0:SIGMOID:GPU), head.cls_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.reg_convs.2.1.conv:0:CONVOLUTION:GPU + head.reg_convs.2.1.bn:0:SCALE:GPU + PWN(PWN(head.reg_convs.2.1.act:0:SIGMOID:GPU), head.reg_convs.2.1.act:1:ELEMENTWISE:GPU).weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [W] Weights [name=head.reg_preds.2:0:CONVOLUTION:GPU || head.obj_preds.2:0:CONVOLUTION:GPU.weight] had the following issues when converted to FP16:
[12/30/2022-14:22:41] [TRT] [W]  - Subnormal FP16 values detected.
[12/30/2022-14:22:41] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[12/30/2022-14:22:41] [TRT] [I] Total Host Persistent Memory: 201760
[12/30/2022-14:22:41] [TRT] [I] Total Device Persistent Memory: 298496
[12/30/2022-14:22:41] [TRT] [I] Total Scratch Memory: 0
[12/30/2022-14:22:41] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 51 MiB, GPU 564 MiB
[12/30/2022-14:22:41] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 42.7218ms to assign 8 blocks to 155 nodes requiring 23483396 bytes.
[12/30/2022-14:22:41] [TRT] [I] Total Activation Memory: 23483396
[12/30/2022-14:22:41] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +17, GPU +32, now: CPU 17, GPU 32 (MiB)
[12/30/2022-14:22:41] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +22, now: CPU 17, GPU 54 (MiB)
[12/30/2022-14:22:41] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[12/30/2022-14:22:41] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
2022-12-30 14:22:41.761 | INFO     | __main__:main:64 - Converted TensorRT model done.
[12/30/2022-14:22:41] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[12/30/2022-14:22:41] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
2022-12-30 14:22:42.089 | INFO     | __main__:main:72 - Converted TensorRT model engine file is saved for C++ inference.
```