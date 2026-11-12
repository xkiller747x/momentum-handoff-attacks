# 方案：第四源模型 Inception-v3（dev276，seed20261101）
目的：FFT/AaF 论文的主要源模型包括 Inc-v3；补上这一源检验“清零代价”和“权衡”的普遍性。只用 dev，不再使用保留集（已按协议用过一次）。
设置：torchvision Inception-v3 IMAGENET1K_V1（transform_input=True，官方权重哈希前缀 0cc3c7bd），引导层 Mixed_6b（FFT 作者设置）；目标 RN50V1、DN121、VGG16-BN、ViT-B/16。
矩阵：base 200 步（160/170/200，含动量）；FFT/ILA/AaF 各自的作者式与继承版；logit-fft 作者式与继承版（清零代价与噪声基线）。
工程检查（8 张 dev 图）：干净样本 8/8 正确，计数与计划一致，模型状态不变，约 0.07 秒/步。
