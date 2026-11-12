# 方案：第三源模型VGG16-BN，检验“重启代价/掌舵权衡”的普遍性

## 背景
conflict reset在两源上均失败并已停止（冲突判定即使16次平均仍≈45–47%，见results/conflict_reset_dev276_v1/report.md）。已确立的发现是：作者式微调重启有约9–10pp代价，同时增强引导掌舵；不重启在CNN目标上不差于作者版、引导收益小时更好、ViT上一致更好。这一发现目前只有两个源模型，按研究准则需要更多源来支撑普遍性。部分衰减等新方法需User确认，本次不做。

## 设置
- 源：torchvision VGG16-BN IMAGENET1K_V1（与FFT/AaF作者主脚本相同权重），原299输入；引导层features[33]（作者vgg16bn.py的l3返回值，即Conv4_3所在block的输出）。权重从官方地址下载，文件名哈希前缀6c64b313与SHA256一致。
- 目标：RN50V1、DN121、ViT-B16；VGG16-BN为源复核。
- 矩阵（dev276，seed20261101）：base 200步（保存160/170/200及动量）；作者FFT；作者logit-fft（重启代价）；FFT continue；logit-fft continue（噪声基线）。AaF视时间。
- 先2图工程检查（严格确定性能否运行、计数、显存、单图耗时）。

## 判断
与前两个源同口径比较：作者FFT vs base200；logit-fft vs base170；FFT vs logit-fft（重启下引导收益）；FFT cont vs logit-fft cont（不重启下引导收益）；FFT cont vs 作者FFT。若三个源一致呈现“重启代价约-10pp、不重启不差于作者版”，则该发现可作为论文核心现象之一。
