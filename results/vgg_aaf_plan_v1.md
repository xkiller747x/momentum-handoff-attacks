# 方案：VGG16-BN源的AaF（seed20261101，dev276）
AaF目前只有RN50（两seed）与DN121（单seed）。补VGG源：作者AaF、AaF continue、logit-aaf continue（同步数同协议对照）；与已有base170/base200评价比较（base175未保存，主比较用同步数的logit-aaf continue与论文口径base200）。代码有改动，base160需重新生成并核对与results/vgg_source_dev276_v1/base的160 checkpoint逐位相同（含动量）。约2.9小时。
