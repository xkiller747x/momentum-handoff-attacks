# 只清冲突动量（conflict reset）：dev276结果（seed20261101）

协议：float起点、继承base160动量、步长2、RDI、L1，第一步把与该步梯度符号相反的继承动量清零后照常累加。base160用新代码重新生成，与原结果（含动量）逐位相同276/276。

## RN50V1源（目标DN121、ViT），成功数/276
|方法|RN50V1|DN121|ViT|
|---|---|---|---|
|base170 / base200|270/270|215/217|4/6|
|作者FFT|266|208|8|
|FFT continue|270|223|9|
|logit-fft continue|269|216|5|
|**FFT conflict**|264|187|4|
|**logit-fft conflict**|269|180|2|

DN121配对：logit conflict vs base170 -12.7pp [-17.0,-8.3]（3/38，log-odds 22/254）；FFT conflict vs 作者FFT -7.6pp [-11.2,-4.0]；vs FFT continue -13.0pp [-17.4,-9.1]；vs base200 -10.9pp [-15.9,-5.8]；FFT conflict vs logit conflict +2.5pp [-2.5,+7.6]（log-odds 154/122，p=.06）。ViT：FFT conflict相对continue log-odds 77/199（更差）。

冲突像素比例：每图46%–50%（中位48.6%）。

## 直白结论（RN50源）
1. 只清冲突动量明显比作者版和不重启版都差，甚至比“只换协议的重启”代价更大。
2. 原因是冲突判定几乎是随机的：单步梯度（一次随机DI抽样）的符号与160步累积动量只在约一半像素上一致，等于随机清掉一半历史；被清的像素随后以2byte步长被噪声梯度驱动，打乱了已有的扰动结构。
3. 这同时揭示了重启代价的来源：单步梯度方向与长期积累方向几乎不相关，清零后头几步的方向主要是噪声。
4. 方向作为“逐像素按单步梯度判断冲突”在本设置下不成立。若要继续，冲突判定必须基于去噪后的方向（例如多次DI平均的引导梯度），但这会增加成本，且与已有的Cautious/逐坐标restart等方法更接近。

DN121源结果待补。

## DN121源（目标RN50V1、ViT），成功数/276
|方法|DN121|RN50V1|ViT|
|---|---|---|---|
|base170 / base200|272/272|146/147|9/4|
|作者FFT|265|181|4|
|FFT continue|269|178|8|
|**FFT conflict**|265|161|3|
|**logit-fft conflict**|271|117|1|

RN50V1目标配对：logit conflict vs base170 -10.5pp [-14.5,-6.9]；FFT conflict vs 作者FFT -7.2pp [-11.2,-3.6]（5/25）；vs FFT continue -6.2pp [-10.1,-2.2]；vs base200 +5.1pp [+0.4,+9.8]；FFT conflict vs logit conflict +15.9pp。ViT：FFT conflict相对continue log-odds 81/195（更差）。冲突比例中位48.4%（45%–50%）。

## 两源合并结论
单步梯度判定冲突的conflict reset在两个源上都显著劣于作者版与不重启版（-7.6/-7.2pp vs作者，-13.0/-6.2pp vs continue），冲突比例都约48%，判定接近随机。下一步按results/conflict_reset_denoised_plan_v1.md做去噪判定诊断（dev64，K=4/16）。
