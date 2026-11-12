# VGG16-BN源的AaF（seed20261101，dev276）

base160与原VGG运行逐位相同（含动量）276/276；全部调用exit0。

|方法|VGG|RN50V1|DN121|ViT|
|---|---|---|---|---|
|base170 / base200|254/254|80/78|93/97|0/0|
|作者AaF|263|87|94|0|
|AaF continue|256|95|119|0|
|logit-aaf continue（同步数对照）|255|78|98|0|

|比较（pp，95%配对bootstrap）|RN50V1|DN121|
|---|---|---|
|作者AaF vs base200|+3.3 [-0.4,+7.2]|-1.1 [-5.4,+3.3]|
|AaF continue vs base200|+6.2 [+2.9,+9.8]|+8.0 [+4.3,+11.6]|
|AaF continue vs 作者AaF|+2.9 [-0.7,+6.5]|+9.1 [+5.1,+13.0]|
|AaF continue vs logit-aaf continue|+6.2 [+3.3,+9.4]|+7.6 [+4.0,+11.2]|
