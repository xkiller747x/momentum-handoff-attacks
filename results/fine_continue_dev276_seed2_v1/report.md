# 第二seed（20261102）dev276复现：作者FFT vs 不重启FFT

全部调用exit0。设置同seed20261101（results/fine_continue_dev276_v1/report.md），base只跑到200步。

|方法|RN50V1|DN121|ViT|
|---|---|---|---|
|base170 / base200|273/272|213/217|3/6|
|作者FFT|268|207|3|
|logit-fft continue（噪声基线）|274|216|6|
|FFT continue|274|226|8|

DN121：作者FFT vs base200 -3.6pp [-7.2,0.0]（log-odds 109/167，p=.0006）；噪声基线vs base170 +1.1pp [-0.7,+2.9]（n.s.）；FFT cont vs base200 +3.3pp [+0.4,+6.2]（13/4，McNemar p=.049；log-odds 175/101，p=1e-5）；FFT cont vs 作者FFT +6.9pp [+3.3,+10.9]（24/5，p=.0006）。ViT：FFT cont相对各对照log-odds均显著为正，成功8/276。

结论与seed1一致且略强：作者式FFT在两个seed上都不优于论文口径的base200（-3.3、-3.6pp）；不重启后在两个seed上都为正（+2.2、+3.3pp），seed2区间排除0。

## AaF第二seed（补跑，2026-11-03下午）

|方法|RN50V1|DN121|ViT|
|---|---|---|---|
|作者AaF|267|215|3|
|logit-aaf continue|272|214|4|
|AaF continue|273|223|4|

DN121：作者AaF vs base200 -0.7pp [-4.3,+3.3]（seed1 +0.7）；AaF cont vs base200 +2.2pp [-1.1,+5.4]，log-odds 186/90（p=8e-9）（seed1 +2.9）；AaF cont vs logit-aaf cont（同步数同协议、只差目标）+3.3pp [+0.7,+5.8]，log-odds 201/75；AaF cont vs 作者AaF +2.9pp [-0.4,+6.5]。ViT成功3-4/276，log-odds上AaF cont相对作者版和logit对照为正。

注意：seed2未保存base175，logit-aaf continue（15步）只能对比base170（10步），其log-odds 169/107为正主要反映多出的5步，不能当噪声基线；本seed以同步数的AaF cont vs logit-aaf cont作为主要证据。
