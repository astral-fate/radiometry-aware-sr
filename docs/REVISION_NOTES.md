# Camera-ready revision: TIME-E 2026, paper #1571363006

- **Edited source:** `conf.tex`. The submitted version is preserved as `conf_submitted_backup.tex`.
- **Compiled PDF:** `build/conf.pdf`, 7 pages (was 6).
- **Figures:** `fig_psnr_paradox.pdf` and `fig_perceptual.pdf` were regenerated from `satalite paper/paper_sr_v2/results.json`. The originals are in `orig_figs/`.

## Corrections found while revising (not raised by the reviewers)
1. **The matching description did not match the code.** The paper said a single affine map, fitted on bicubic, was applied to both outputs. The code (`flair_sr_modal.py` L447-448) fits the SR output and bicubic to the target *separately*. The method text now describes the code: a symmetric oracle control, where each output gets its own optimal fit, plus the correlation-based interpretation. No numbers change.
2. **Stale numbers in the prose now match `results.json`, which the figures are built from:**
   - Raw gain "up to 15.23" is now **15.01** dB.
   - The S2→SPOT range is now **14.70–15.01** dB. The SwinIR mean of 10.39 is lower because one seed diverged.
   - SPOT→aerial is now **6.98–7.29** dB.
   - S2→SPOT sharpness: SwinIR **0.91×** (was 1.11), EDSR **1.00×** (was 1.02), RCAN **0.65×**, SRCNN **0.59×**.
   - SPOT→aerial sharpness: EDSR **3.22±0.27×** and SRCNN **1.90×**. The nDSM table keeps its paired runs (3.08×), and a note in the text explains the difference.
3. **Eq. (5) and the residual formula now match the code.**
   - The gradient is now the mean absolute forward difference.
   - Sharpness S is defined explicitly.
   - The text notes that EDSR upsamples internally and that DDPM is not residual.
4. **Fig. 2 and Fig. 3 legend fixes:**
   - Removed the stale "hatched = 256-tile subset, own reference". The DDPM in `results.json` is evaluated on the full 1,332-tile set against the shared control.
   - Removed the uncited "±0.1 dB on which published methods differ" band.
5. **The nDSM zero-init sentence has been corrected.** The backbone is trained jointly from scratch, not frozen.

## Reviewer 1
| Comment | Change |
|---|---|
| Justify the affine model | §III: new paragraph covering the first-order gain/offset model, closed-form fit, and the fact that it cannot add spatial structure |
| State its limitations | §III: spectral response, nonlinearity, spatially varying illumination and acquisition time |
| Evaluation-only control; target information used | §III: "oracle evaluation control", never used in training or inference, not a preprocessing step |
| Estimated per test image? Same for all configs? | §III: yes to both, stated explicitly |
| Raw−matched is not a decomposition | §III and §V: now described as an empirical diagnostic, not an exact decomposition |
| Absolute PSNR | New Table II: raw and matched PSNR, LPIPS and S for every model and configuration |
| Define LPIPS improvement | §IV and Fig. 3 caption: LPIPS(bic) − LPIPS(SR); AlexNet; inputs in [−1,1] |
| Reproducibility | New §IV Experimental Setup |
| Method vs. evaluation | End of the Introduction: the contribution is an evaluation methodology, not a new SR network |

## Reviewer 2
| Comment | Change |
|---|---|
| Train/test counts, patches, patch size, channels | §IV: 4,662 / 1,332 tiles; 10/64/512 px; RGB bands; normalization |
| Loss, optimizer, LR, epochs, data, retraining | §IV: L1 or DDPM MSE; AdamW at 2e-4 constant; effective batch 32; 25 epochs; same tiles for every model; retrained from scratch for each sensor pair; no validation split |
| Define sharpness and its limits | §III: definition, why it is a proxy, and the fact that it measures amount, not correctness (ringing, noise, hallucination) |
| nDSM as a focused case study | §V, Abstract, Introduction and Conclusion: SPOT→aerial, SRCNN and EDSR only |
| Cautious Abstract | Abstract now reads "−0.18 → +0.02 dB, approximate parity" |
| Uncertainty | §IV: ± is the standard deviation over 5 seeds; single-seed cells are marked †; nDSM spread is discussed |

## Reviewer 3
| Comment | Change |
|---|---|
| Absolute values for relative claims | Table II, plus absolute context for 5.55× (about ¼ of target G) |
| Land-cover masks and aggregation | §V: FLAIR labels on the 0.20 m grid; 300 tiles; pixel-pooled; top 8 classes, each ≥0.5% (more than 1.7M px); no per-class uncertainty, so results are indicative |
| nDSM not uniformly beneficial | §V and Conclusion: plowed land decreases; effect is class- and scene-dependent |
| Related-work comparison table | New §II plus Table I |
| PSNR, ΔPSNR, matched ΔPSNR; ±; notation; sharpness vs. gradient | Definitions in §III and §IV; "S2" defined in the Table II caption |
| Abstract matches Results | 15.01 dB and nDSM wording updated |

## Still to check
- **Table I checkmarks** for prior work, especially OpenSR-test "spatial-detail metric" (its *synthesis* metric). A co-author should confirm them.
- **Fig. 6** (second qualitative example) was removed to stay at 7 pages.
- **Not reproducible from local files:** per-class pixel counts (derived from shares), per-class uncertainty, GPU types and absolute PSNR for the nDSM runs. These would need the `flair-sr-512-runs` logs from HF.
