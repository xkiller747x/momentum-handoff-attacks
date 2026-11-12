# FTM read-only reference fixtures

Author: uhiu/feature-tuning-mixup, commit
`912b8da37288ad1c339bf974408ca20d4e9bdf3d`.

| Local fixture | Pinned original | SHA256 |
|---|---|---|
| ftm_attacks.py.txt | https://raw.githubusercontent.com/uhiu/feature-tuning-mixup/912b8da37288ad1c339bf974408ca20d4e9bdf3d/attacks.py | 4673e39590d3ad362563e4b90232b34bc8df3e5bc9df66b50d0819c2ade6e8ad |
| ftm_config.py.txt | https://raw.githubusercontent.com/uhiu/feature-tuning-mixup/912b8da37288ad1c339bf974408ca20d4e9bdf3d/config.py | 84863e7c1104e5c7f266be42068dd2361e4820432ba1a82e8b7b1ee222493665 |
| FTM-LICENSE.txt | https://raw.githubusercontent.com/uhiu/feature-tuning-mixup/912b8da37288ad1c339bf974408ca20d4e9bdf3d/LICENSE | 09a7d7e49842d5a57184f2ca2399ff7a37561c403f3c45dd2788720a988da70d |

MIT license, Copyright(c)2025 uhiu. The first two fixtures are byte-for-byte
copies of the root's read-only source snapshots; license was fetched from the
same pinned commit. Only selected original AST definitions are executed in the
test namespace. No upstream imports, dataset loaders, main, model downloads,
or GPU initialization execute. Normal-PDF dependency and empty_cache are test
stubs; observation subclass only snapshots perturbations, preserving original
hook/forward/update code. CPU global author RNG and dedicated adapter RNG start
from the same seed; final states must match. Native224 is compared to native
author semantics. mapped299 is tested separately against explicit boundary
shapes, not incorrectly equated with native224.
