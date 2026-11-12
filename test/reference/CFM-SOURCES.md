# CFM read-only reference fixtures

Author repository: https://github.com/dreamflake/CFM

Pinned commit: `a402db042d997d9f32b9a3967b58fe2b386f1b2d`.
These are byte-for-byte copies of root's primary source snapshots, recorded in
`results/targeted_feature_sources_20261028/cfm_source_provenance_v2.json`.

| Fixture | Pinned original | SHA256 |
|---|---|---|
| cfm_attacks.py.txt | https://raw.githubusercontent.com/dreamflake/CFM/a402db042d997d9f32b9a3967b58fe2b386f1b2d/attacks.py | 89e856d6292fdace66952f87c11e4952c5ddbdc35b7dadbcee0964b45b16276e |
| cfm_config.py.txt | https://raw.githubusercontent.com/dreamflake/CFM/a402db042d997d9f32b9a3967b58fe2b386f1b2d/config.py | bf60018b8d6ffbd67c2e5c9d9a9f4c3a18fb937edd401a71c4409c8349fd95cf |

Only the original `FeatureMixup` AST class and literal config578 are used in
the isolated CPU oracle. Upstream imports/PyTorch3D/main/model loaders do not
execute. The test maps hard-coded `Tensor.cuda()` transfers to CPU identity;
mixing arithmetic and torch random draw order are unchanged. Probability is
overridden only for the declared p0/.1/1 tests. Native-size small-model oracle
checks use zero tolerance for forwards, input gradients, final pixels and CPU
RNG. Composite TI/RDI/projection use the same already-audited local primitives,
so this is a pure FeatureMixup comparison, not an independent full-script
reproduction. The RN50 meta-only check separately validates original recursive
module order and exact299 eligibility, without real weights or image data.

No explicit license is established by these snapshots; local private read-only
fixture storage is not a redistribution-permission or MIT-license claim.
