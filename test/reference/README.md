# Fixed upstream reference fixtures

These UTF-8 text files are read-only oracles in `test_related.py`. Tests AST-load
only original function/class definitions with minimal base-class stubs, excluding
imports and preventing model loading. They never import an upstream package.

| Fixture | Commit and upstream path | SHA256 |
|---|---|---|
| bsr.py.txt | Trustworthy-AI-Group/TransferAttack `ae960f546a907a193931a2df12afd0f3fef2e0a5`, `transferattack/input_transformation/bsr.py` | `7f054309c25616c66a581c4497a7a1841e41b949699d4b6baaa0e227cbfe0a05` |
| ops.py.txt | the-full/OPS `7914115d86d1add0f09c3a0720d12961e14a0c25`, `TransferAttack/transferattack/input_transformation/ops.py` | `b25bafbac3c5e008f351f40612ce50d981f9fea689b5d40a3dc5dc31084ae231` |

Both repositories carry MIT licenses. Exact pinned root licenses are included as
`TransferAttack-LICENSE.txt` (copyright2023 Trustworthy-AI-Group) and
`OPS-LICENSE.txt` (copyright2025 include"five.h"). Sources:

- https://github.com/Trustworthy-AI-Group/TransferAttack/blob/ae960f546a907a193931a2df12afd0f3fef2e0a5/LICENSE
- https://github.com/the-full/OPS/blob/7914115d86d1add0f09c3a0720d12961e14a0c25/LICENSE

Tests mock actual cut lengths, permutations, angles, shifts, and DI integer
choices. Matching global seeds across Python/NumPy/torch alone is not used as a
differential oracle. Fixed-noise pool construction is compared separately using
the same explicit CPU torch generator seed and uniform call shapes.
