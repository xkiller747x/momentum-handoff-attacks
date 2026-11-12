# 方案：DN121与VGG16-BN源的第二seed（20261102），dev276
目的：三源“重启代价/不重启不差于作者版”的结论中，DN121与VGG只有单seed。补第二seed后三源均有两seed。
矩阵（每个源）：base 200步（保存160/170/200及动量）；作者FFT；作者logit-fft；FFT continue；logit-fft continue；评价base170/200与4个微调。目标：DN121源→RN50V1/ViT；VGG源→RN50V1/DN121/ViT。代码不改动。约4.5小时。
