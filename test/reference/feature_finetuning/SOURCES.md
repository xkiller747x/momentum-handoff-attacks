# Fixed FFT/AaF CPU oracle evidence

These `.source.txt` files are read-only source evidence copied from the supplied
`results/targeted_feature_72h_sources_v1` snapshot. Their exact URLs, commits,
file sizes and SHA256 values are in `fft_aaf_source_provenance_v1.json`.
FFT: zengh5/TA_feature_FT `543d258e19e35646ff68d9f4b933f3a4eec2d076`.
AaF: zengh5/Avg_FT `9b692d127f38d0d404c2be86e8657e6176b34eb3`.
No license/redistribution grant is inferred from source availability.

Tests extract original classes/functions with AST, omit imports and all caller
entrypoints, replace only CUDA tensor transport with CPU, and supply the normal
PDF formula instead of importing scipy. Read-only iteration instrumentation
copies the author's existing local tensors after each iteration and at return.
The actual parameter-backward author attack executes unchanged mathematical
statements, separately from local frozen-parameter `autograd.grad`.
Draws/masks use exact comparisons; forward/backward tensor comparisons disclose
atol2e-6/rtol2e-5. This does not claim CPU/GPU bitwise equivalence.
