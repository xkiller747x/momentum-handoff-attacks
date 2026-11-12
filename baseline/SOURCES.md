# Baseline implementation sources and profiles

The local code is a small adaptation, not an execution of upstream packages.
It implements untargeted L-infinity attacks without random starts. All models
consume the same Pillow bilinear 224x224 uint8 canonical PNG; only normalization
occurs inside each model wrapper. This is not original-paper numerical replication.

- MI and momentum/update equations: TransferAttack commit
  `ae960f546a907a193931a2df12afd0f3fef2e0a5`,
  [attack.py](https://github.com/Trustworthy-AI-Group/TransferAttack/blob/ae960f546a907a193931a2df12afd0f3fef2e0a5/transferattack/attack.py)
  and [mifgsm.py](https://github.com/Trustworthy-AI-Group/TransferAttack/blob/ae960f546a907a193931a2df12afd0f3fef2e0a5/transferattack/gradient/mifgsm.py).
- Momentum DI uses the same commit's
  [dim.py](https://github.com/Trustworthy-AI-Group/TransferAttack/blob/ae960f546a907a193931a2df12afd0f3fef2e0a5/transferattack/input_transformation/dim.py):
  probability .5, rate 1.1, integer resize sampled from [224,246), zero padding
  to246 then bilinear resize back224; momentum1. DI-N independently samples N
  such views per iteration and averages raw input gradients before normalization.
- SIA follows author SIT commit `6d5432d4165a85753c8ae8fdee5cfd3362509a97`
  [attack.py](https://github.com/xiaosen-wang/SIT/blob/6d5432d4165a85753c8ae8fdee5cfd3362509a97/attack.py),
  with random 3x3 cut points on every view and 10 uniformly selected block
  operations: resize, two shifts, two flips,180-degree rotation, intensity scale,
  fixed16/255 noise, DCT truncation, dropout2d(.1). The cited TransferAttack
  [sia.py](https://github.com/Trustworthy-AI-Group/TransferAttack/blob/ae960f546a907a193931a2df12afd0f3fef2e0a5/transferattack/input_transformation/sia.py)
  has7 operations and is not this profile. Unused author Gaussian kernels are
  omitted. Author rectangular resize intermediate axis order and DCT `-0:`
  slicing for tiny blocks are retained. DCT output is not clipped; dropout scales
  by1/.9. Neither final transformed view is globally clipped.

Random number streams use reusable per-view seeds, Python RNG for discrete
choices and CPU torch RNG for continuous samples/dropout. Sampling distributions
match the described operations, but random streams do not match upstream NumPy/
torch global seeds bit-for-bit. Concatenation replaces in-place block assignment
to preserve differentiability. Model parameters are frozen, eval/FP32; TF32 is
disabled by the CLI. Each microbatch's summed CE is divided by total views.
Raw gradient mean normalization has a1e-12 floor; zero and near-zero occurrences
are counted. MI/DI require one view, DI-N/SIA views must be explicit in commands.

Only actual surrogate images passed to model forward/input-gradient computation
are counted; no implicit prediction/evaluation is performed by attack. Pixel
counts refer to model-input spatial pixels (channels excluded). Per-image attack
time includes transformations and gradients, excludes PNG writing and model load.
CUDA peak allocated/reserved is reset per image and includes resident model.

## Recent baseline profiles added in isolated staging

BSR uses **bsr-transferattack-3blocks** at TransferAttack
`ae960f546a907a193931a2df12afd0f3fef2e0a5`,
`transferattack/input_transformation/bsr.py`: constructor line35 (N20,3blocks),
get_length lines41–45, shuffle_single_dim47–51, BILINEAR RandomRotation53–55,
and directional strip shuffle57–61. Legacy `np.random.uniform(2,size=3)` is
implemented as `2-U` (U uniform[0,1)), then NumPy half-even round and first-argmax
correction. The first direction is randomized. Each reordered first strip gets
an independent angle and independent second-direction cut/ordering. No regular
9-cell grid is substituted. Rotation/fill0 operates in[0,1] before normalization.
N26 is explicitly a gradient-image-cost-matched variant, not this default point.

This is distinct from original BSR TensorFlow commit
`6ec37d9ef176cb962f6cc062733bf8dc118627b6`,
[mi_bsr.py](https://github.com/Trustworthy-AI-Group/BSR/blob/6ec37d9ef176cb962f6cc062733bf8dc118627b6/mi_bsr.py):
2blocks/N20 lines59–60; max_angles=.2 line62; input[-1,1] line111;
Uniform(0,1) length weights lines150–156; global two-direction cuts and
per-cell truncated-normal rotations std=.2 radians in shuffle_rotate169–179;
the TensorFlow-addons default fill0 there corresponds to gray.5 in[0,1].
Clean source argmax labels are fed to the attack at257/261, unlike this project's
explicit true labels. Original TensorFlow numerical replication is not claimed.

OPS uses **ops-author-31ops-93pool**, author commit
`7914115d86d1add0f09c3a0720d12961e14a0c25`,
[TransferAttack/transferattack/input_transformation/ops.py](https://github.com/the-full/OPS/blob/7914115d86d1add0f09c3a0720d12961e14a0c25/TransferAttack/transferattack/input_transformation/ops.py).
DI35–60 always applies, resizes back to original width×width (thus canonical224
network inputs stay224). Fixed rotates71–76 use torchvision default NEAREST/fill0.
The31 basic operators111–117 consist of identity, two flips, two shifts,
angles5,-5,15,-15,45,-45,90,-90,180; integer divisions2..8; DI rates1.1..2.9
in.2 increments. There are no extra quantization/median operators.

Composite reduce132–136 executes the last selected basic function first.
Pools142–148 contain31 independent composites at each level2,3,4, sampling
basics with replacement, for93 selectable function instances. A new pool is
built for each selected neighbor. Noise pools152–163 are6 radii × Ne and fixed
for an entire image attack; every iteration samples Ne without replacement.
Mean gradient165–184 includes the untransformed base gradient and Ne×Np extra
gradients, with no neighborhood clipping. Main Ne=Np5, beta2,T10 therefore260
gradient images. Zero Ne or Np produces MI. Zero radius alone does not remove
operator effects. Local sampling is distributional, not upstream RNG-bit identity.

OPS `pool_init_seconds` is inside `elapsed_seconds`, including fixed-pool setup.
New-method records also contain transformed range/out-of-range channel-element
counts. No arbitrary out-of-range threshold triggers clipping or rejection.
NumPy version and torchvision version are saved with run configuration/identity.

### OPS paper/code lifecycle difference verified 2026-10-27

The original CVF PDF, page4 (printed19178), Algorithm1 lines2–4, initializes
the perturbation set inside the outer iteration loop. This was verified from
the rendered page, not only text extraction. The pinned author implementation
above calls `init_eps_list` before that loop, keeping the pool fixed. The local
`ops-author-31ops-93pool` follows the latter and is not a reproduction of the
paper's pool lifecycle. Neither pool refreshing nor this discrepancy is claimed
as a new attack method. Existing frozen runs remain unchanged. The separately
labeled lifecycle adaptation is `ops-author-noiserefresh`; validation and results
follow `results/ops_profile_protocol.md` and are tracked in `record/paper.md`.
The original20 completed before its code was merged. PDF SHA256:
`6cdab6f125d0fb1505085a18e57b4525c40e60366714a81a7e32bcd2a014b66c`.
The PDF and rendered page are in `results/literature_ops_verification/`.

The paper's page5, section4.2, also specifies composition orders[1,2,3,4],
whereas the pinned author's default and this profile use[2,3,4] (93 instances).
The lifecycle-only `ops-author-noiserefresh` supplement deliberately retains
the latter and is not called a full paper profile. No level1 branch is added
to the frozen comparison. The same section specifies Ne=Np5,T10,epsilon16,
alpha1.6,mu1, attack batch32; this project's epsilon8/RN50-V2/224/per-image
protocol remains a separate explicitly documented adaptation.

Pinned BSR/OPS text fixtures, SHA256 and MIT licenses live under
`test/reference/`; AST extraction tests never execute upstream model loaders.
Explicit cut/angle/shift/DI plans are mocked for output and raw-gradient
differentials, reporting absolute and relative L2 error. Non-square OPS DI follows
its upstream square-by-width output; mixed non-square microbatches are outside
the canonical224 protocol and are not padded into a newly defined algorithm.

## Known per-view L2 control

`ops-view-l2` / `ops-fresh-perview-l2-wrtx-control` is a local known-principle
control on the unchanged OPS fresh sampler. It normalizes each original-input
gradient before averaging; it is not a complete SimAttack implementation.
[SimAttack's official manuscript](https://openreview.net/pdf?id=FWVTbp54LQ),
indexed section3.3 and Algorithm1, already describes per-input L2 followed by
aggregation and momentum. The indexed algorithm's differentiation variable was
verified on2026-10-27: it is the adversarial input before transformation.
The manuscript's transformation set and default sampling count differ from this
local26-view OPS control. Full author code/sampling details and publication
status remain unverified; actual reading scope stays in `record/paper.md`.

This local control completed engineering and a24-image fixed-point source probe,
with no formal target comparison in the original twelve-hour window. A separately frozen two-hour continuation subsequently completed the paired three-seed target comparison; see results/view_l2_continuation_report_v1 and record/paper.md. Protocol and source/data hashes
are in `results/view_l2_control_protocol.md` and the corresponding run configs.

## Targeted AA/GAA/PAA and logit baselines: local224 adaptation

The first targeted batch lives in `baseline/targeted.py` and
`utils/targeted_data.py` / `utils/targeted_runner.py`. AA-style flatten-L2 follows
[Feature Space Perturbations Yield More Transferable Adversarial Examples,
CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Inkawhich_Feature_Space_Perturbations_Yield_More_Transferable_Adversarial_Examples_CVPR_2019_paper.html).
The actual AA/GAA/PAA source inspected is the PAA/GAA authors' fixed
[fp_attack_res50.py](https://github.com/yaya-cheng/PAA-GAA/blob/2697842ecc8a1c6d6ae214bd18d951a8bc05d89f/fp_attack_res50.py)
at commit `2697842ecc8a1c6d6ae214bd18d951a8bc05d89f`, SHA256
`8f135e1c3c30ff2415ff8f86fec55e9edde6c03f85a7c8ba16368ad638a4de5f`.
Original read-only text and README provenance are in the main project's
`results/targeted_feature_sources_20261028/`; upstream main/model-loading code is
not imported or executed here. GAA implements the author's population **std**,
not the paper's variance wording: squared channel mean/std discrepancy×HW/C².
Centered-vector norm/sqrt(HW) preserves the forward formula and defines zero-std
derivatives as zero without epsilon. AA uses the norm, not squared Euclidean.
PAA-poly uses exact c0,d2 polynomial MMD, equivalently squared channel-Gram
discrepancy/(4H²W²C²). Its exact candidate selection explicitly differs from the
author's random linear-time selection approximation. All use their own maximum
distance for `farthest`, with reference-ID ties resolved first.

Profiles are `aa-l2-plain-mi-local224`, `gaa-mean-std-plain-mi-local224` and
`paa-exact-poly-c0-d2-plain-mi-local224`. They use ImageNetV2 MatchedFrequency's
10 target-class references, not the authors' ImageNet-training20-image gallery;
source RN50 ImageNet1K_V2 and layers3/4; shared full-image bilinear224; explicit
Linf projection and alpha-bytes; plain L1-normalized MI descent, mu1, without
feature RDI/TI. Zero gradients remain zero. Local299-gradient/epsilon16 and
alpha16/299 or2bytes are predeclared adaptations, not original-paper numerical
reproduction. First/index/nearest rules are labeled controls, not new selectors.

`target-logit-plain-mi` maximizes the specified target logit.
`target-logit-rdi-nearest-ti5-mi` uses the non-feature RTM backbone from
[FTM attacks.py](https://github.com/uhiu/feature-tuning-mixup/blob/912b8da37288ad1c339bf974408ca20d4e9bdf3d/attacks.py),
commit `912b8da37288ad1c339bf974408ca20d4e9bdf3d`, SHA256
`4673e39590d3ad362563e4b90232b34bc8df3e5bc9df66b50d0819c2ade6e8ad`.
RDI retains resize/left/top draw order, nearest interpolation and340/299 ratio;
a dedicated per-image CPU generator replaces author-global RNG. TI uses5×5
normalized normal-PDF samples on[-3,3]. These profiles contain no FTM feature
tuning. All references/clean/optimization/checkpoint-float/checkpoint-PNG
forwards are counted independently of input gradients. Evaluation reloads saved
PNG and uses specified-target argmax, with full and conditional denominators.

## FTM author-config baseline adaptation

`baseline/ftm.py` / `utils/targeted_ftm.py` add FTM from the same fixed
uhiu/feature-tuning-mixup commit `912b8da37288ad1c339bf974408ca20d4e9bdf3d`.
Pinned attacks/config text, SHA256 and MIT license are in
`test/reference/FTM-SOURCES.md`. Only pure original AST definitions serve as CPU
oracles; upstream main/model/data loaders are never imported. Authors' config
uses divisor4 and `conv_linear_include_last`, unlike the paper-depth16 or
no-last alternatives. This baseline keeps C/SelfShuffle/channelwise/M, ratio
0–.75, probability.1, beta.01, lastFC included. Selected feature perturbations
descend the sum target-logit gradient, input ascends; unselected layers still add
detached learned perturbations. Scale norms are per image, detached, with the
author's1e-7 perturbation-norm denominator. Zero input gradient is explicitly
mapped to zero rather than author zero/zero; positive-gradient math is unchanged.

Primary profile `ftm-author-config-mapped299-local224-batch2` records every
Conv/Linear's actual299 zero-input output shape, selects Linear or width≤74,
then maps those exact module paths onto224 inputs. It does not guess entire
ResNet stages. A native224 optional profile uses the original width≤56 rule;
the299/224 eligibility difference is explicit. Qualification adds one separately
counted source forward per invocation (including resumed invocations), its pixel
cost/time/peak and layer-table SHA. Native224 also records a separately counted
224 qualification forward for a frozen explicit layer identity.

Inputs are local224/RN50V2, not author299/V1. Batch2 uses sorted adjacent cohort
IDs with fixed pair clusters; batch1 SelfShuffle self-mixes. randperm may be
identity; no forced partner switch is added. Each pair uses one dedicated CPU
generator for RDI and all hook draws in author order.299 means input-gradient
steps plus one separate clean-feature cache call per batch, not total source
queries. Clean and float/PNG endpoints fully disable FTM without advancing RNG
or mutating cache/perturbations. Per-image costs share batch counts/time rather
than duplicating batch totals; qualification and failed observed attempts remain
separate. Generic target evaluation loads clean model instances and preserves
pair/cluster metadata; existing point-estimate reports introduce no image-IID CI.
These are declared local adaptations, not a numerical reproduction of the
original paper or a new attack mechanism.

The optional `--trace-updates` / `ftm-update-report` is a low-priority replication
audit of the unchanged zero initialization, raw gradient subtraction and
detached scale denominator. Default full mode retains the exact prior update
expression and no observational tensor copies. The separately labeled
`first-nonzero-only` known control forms all gradients/proposals and preserves
RNG/forward/backward count, freezing each sample only after its first actual
nonzero u update. Neither this control nor FP64 CPU observation changes epsilon,
initialization or optimizer, and neither is claimed as a new attack or repair.
Trace distinguishes raw u, FP32 changes, FP64 unit-direction drift and
fixed-current-z effective-v parameterization; Conv and FC are reported separately.
No target-transfer effect or CI is inferred by this source-only audit.

## RaPA CVPR2026: author-hook local224/V2 adaptation

`baseline/rapa.py`, `utils/targeted_rapa.py`, and `utils/rapa_probe.py` use pinned
molarsu/RaPA commit `fa0c4a311d6a5e1bf366c4f29dc84bd283430b0a`.
[Original DropConnect hooks](https://github.com/molarsu/RaPA/blob/fa0c4a311d6a5e1bf366c4f29dc84bd283430b0a/core/attacker/DropConnect.py)
and [MultiGetGradient](https://github.com/molarsu/RaPA/blob/fa0c4a311d6a5e1bf366c4f29dc84bd283430b0a/core/attacker_component/GetGradient/MultiGetGradient.py)
are held as private read-only fixtures with SHA256 in
`test/reference/RAPA-SOURCES.md`. No explicit license has been established;
MIT or redistribution permission is not asserted. No mmengine/mmpretrain,
registry, upstream imports or automatic model loading execute.

The actual RN50 source is torchvision `resnet50(pretrained=True)` (V1), wrapped
only by ImageNet normalization; original NIPS images use ToTensor without a
canonical224 resize, retaining299 for CNNs. Local RN50V2/canonical224,
single-image T299/S1 default is an explicit adaptation, not original
T1000/S5/batch32 numerical reproduction. p.05 follows the RN50 command setting.
All actual Linear/BN2d layers are selected by named_modules order. Weight mask
is Bernoulli(1-p), bias mask is `(rand>p).float()`, neither rescaled. p0 still
samples, including bias exact-zero draws. The original pre-hook `.data` masking
and immediate forward-hook `.data` restoration are retained verbatim in
operation order; no functional/autograd repair replaces these semantics.

Each of S samples redraws RDI and immediately computes its input gradient;
the source code sums gradients (paper Algorithm1 says mean), then one TI5/L1/MI
and image-form local-RTM Linf update runs. Zero L1 maps to zero, explicitly
defining the author's undefined zero/zero boundary. RDI/parameter masks use
independent labeled CPU/device generators; same-device oracle agreement does
not claim cross-device identical RNG. Endpoints are strictly disabled-hook
plain forwards. The semantic probe reports independent Linear and eval-BN
Jacobians and composite direct author-oracle differences; it never assumes the
whole network is necessarily all-masked or all-original in backward. Parameter
restoration/BN preservation and every actual F/B/mask/restore cost are recorded.

## FFT (ICASSP2024) / AaF (ICASSP2025), original299/V1 batch1

FFT [author attack](https://github.com/zengh5/TA_feature_FT/blob/543d258e19e35646ff68d9f4b933f3a4eec2d076/attack_feature_FT.py)
and AaF [author attack](https://github.com/zengh5/Avg_FT/blob/9b692d127f38d0d404c2be86e8657e6176b34eb3/attack_Ave_FT.py)
are pinned together with caller, normalization/DI/RPA and ResNet definitions in
the supplied source snapshot and `test/reference/feature_finetuning` provenance.
No unverified license permission is asserted.

`baseline/feature_finetuning.py` uses fixed layer3, original RGB299 PNG and RN50
IMAGENET1K_V1. The author's initial shape-only query and unused layer1/2/4
guidance tensors are omitted; all actual forwards are counted. FFT normalizes
before Bernoulli masking; AaF masks raw pixels
before normalization, preserves RPA as_strided/concatenation/edge behavior and
smooths aggregate feature gradients. Each original/target guidance uses30
samples and correction1 std. Beta.2, sum inner product ascent, author NumPy DI,
TI5, unnormalized fine momentum, epsilon/16 and AaF15-step preclip feedback are
retained. `logit-fft`/`logit-aaf` are matched optimizer controls without guidance
forwards/backwards; they discard the same60 mask draws to align the subsequent
single-stream DI parameters, recording CPU sampling cost separately.

This is a batch1/local per-image RandomState adaptation, not author batch4/global
RNG or main-table reproduction. Guidance uses frozen-parameter autograd.grad
to layer3 rather than author parameter backward. All source forwards remain
complete; feature-B and input-B are separately reported and are not interchangeable
compute costs. Base generation reuses the existing local torch-RDI logit-RTM.
Independent evaluation uses RN50V1/DN121 normalization at299; ViT directly
bilinear-resizes299 to224 with align_corners=False/antialias=False before weight
normalization. Existing224 results are not numerical comparators for this profile.

The CPU author-attack oracle uses `mask_count=4` (covering pixel/3/5/7 RPA),
while the normal CLI fixes30 samples per class; the30-sample pretrained profile
still requires separate engineering validation. Guidance/iteration floats use
atol2e-6/rtol2e-5, and decoded final PNG pixels are compared exactly. This is
not a claim of bitwise equality throughout floating-point execution.
The driver saves only original/target/combined guidance scalar summaries from
the actual existing observer tensors: correction1 std, FP64-accumulated L2,
nonfinite/zero counts and method meanings. No guidance arrays or new F/B are
introduced. Logit controls explicitly save an empty, inapplicable summary.

