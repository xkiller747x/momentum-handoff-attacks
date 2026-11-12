# Private read-only RaPA oracle fixtures

Author repository: molarsu/RaPA, pinned commit
`fa0c4a311d6a5e1bf366c4f29dc84bd283430b0a`.

| File | Original relative path | SHA256 |
|---|---|---|
| rapa_dropconnect.py.txt | core/attacker/DropConnect.py | 445540161ecf2ec27771e7db19deefa9f3e07d6e970b3f0fafaefbe4e84bd86a |
| rapa_multigradient.py.txt | core/attacker_component/GetGradient/MultiGetGradient.py | 58d840a50e63d3683bb71cb964b37b5f76bb1103415f03285cb80e26fa5abf5a |

Official immutable source URLs:

- https://raw.githubusercontent.com/molarsu/RaPA/fa0c4a311d6a5e1bf366c4f29dc84bd283430b0a/core/attacker/DropConnect.py
- https://raw.githubusercontent.com/molarsu/RaPA/fa0c4a311d6a5e1bf366c4f29dc84bd283430b0a/core/attacker_component/GetGradient/MultiGetGradient.py

Copied only from root's existing source snapshots; this coding batch did not
download anything. No explicit source license has been established. These are
private read-only provenance/oracle fixtures, not a claim of MIT licensing or
permission to redistribute. Check permissions before publishing these files.
AST extraction executes only the original pre/post/selection definitions; no
registry, BaseAttacker constructor, imports, main or pretrained loader executes.
MultiGetGradient testing extracts its original class without its registry
decorator, with a minimal base storing loss/transform attributes. It does not
replace the S-loop implementation.
