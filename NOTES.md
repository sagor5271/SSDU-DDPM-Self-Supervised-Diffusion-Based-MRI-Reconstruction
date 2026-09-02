# Known issues / open design questions

Kept here deliberately, rather than glossed over, so the state of the
code is honestly represented.

## 1. Train/test mask & data-consistency mismatch (unresolved)

`PhysicsUNet.forward()`'s `mask` argument is documented as
"measured/Aleph mask only". This holds during training
(`SSDUDDPM.train_step()` always passes `mask=aleph` and
`measured_kspace=kaleph`, the Aleph-only subset), but at test time
`SSDUDDPM.sample()` passes `mask=omega` and
`measured_kspace=measured_kspace` — the **full** acquired Omega set —
for both the network's mask channel and every internal/final
data-consistency step, not just the final one.

This means the network sees a different mask-channel distribution at
test time (≈100% of non-ACS lines "measured") than at train time
(≈`SSDU_ALPHA` fraction). This may be an intentional choice (many SSDU
papers do condition test-time inference on all available data, since
there is no reason to hold anything back once training is done), but
it has not been explicitly decided/documented as such in this
codebase, and the header comments in earlier versions describe a
"final-DC-only" behavior that the current `sample()` implementation
does not match.

**Before reporting results**, decide and document one of:

- **(a)** Keep `sample()` as-is (condition + DC on full Omega
  throughout the reverse chain) — update the docstring in
  `network.py`/`diffusion_model.py` to say so explicitly, and drop the
  "Aleph-derived condition, full Omega only at final DC" language.
- **(b)** Change `sample()` to condition on an Aleph-derived
  input/mask throughout (matching training), and only substitute the
  full-Omega data-consistency correction on the last reverse-diffusion
  step.

Either is defensible; what matters is that the code and its
documentation agree, and that the choice is called out explicitly in
any writeup.

## 2. `fixed_probability_split_multi` GPU↔CPU round-trip

When `CFG.USE_LEARNED_PARTITIONING = False` (the default), every
training step calls `split_ssdu()` once per (batch, contrast) slice,
which is NumPy-based and requires moving each mask tensor GPU → CPU →
GPU. This is not a correctness bug, but it will slow down training
noticeably at larger batch sizes / more contrasts. A vectorized,
torch-native version of `split_ssdu` would remove this overhead if
training speed becomes a bottleneck.

## 3. Multi-contrast pairing caveat

When `CFG.NUM_CONTRASTS > 1`, contrasts are paired by **slice index
only**, not by patient — i.e. item `i` of contrast A is not guaranteed
to be the same anatomical slice, or even the same patient, as item `i`
of contrast B on the standard FastMRI knee directory layout. Verify
this assumption holds for your data (or replace with proper
patient-matched pairing) before relying on multi-contrast results.
