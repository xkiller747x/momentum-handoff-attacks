# 方案：不重启的特征引导继续（dev64）

## 动机
拆解实验表明FFT/AaF的主要代价是微调时动量清零。若引导确实有用，那么在不清零、继承原攻击动量的情况下只把目标换成feature引导，应当相对“直接继续原攻击”有净收益。

## 做法
- base160重新生成并额外保存动量（--save-momentum），核对图像与原dev64逐位相同。
- continue模式：从base160的float图像起步，继承base动量，使用与原攻击相同的步长2byte、RDI、TI、L1-MI；只把目标换成FFT/AaF的feature引导（AaF还保留其轨迹平均和patch mask）。
- 对照：同模式下的logit目标（logit-fft/logit-aaf continue），它与base170/175只差DI随机抽样流，可作为“随机性本身造成多大波动”的噪声基线。
- 主比较：FFT-continue vs logit-fft-continue（同一切，只差目标），以及 vs base170；AaF同理对175。

## 边界
dev64单seed开发证据；若引导在不重启时有净收益，再上dev276并查文献是否已有人这样做，才可能成为方法候选。不调参，不用保留集。
