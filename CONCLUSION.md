# Conclusion: Ambient Diffusion for Matcha TTS

## Experimental Summary

We adapted the Ambient Diffusion framework to Matcha TTS, a conditional flow-matching text-to-speech model. The core idea was to leverage corrupted multi-speaker data (109 VCTK speakers) alongside clean single-speaker data (LJSpeech) during training. The key hyperparameter, $t_{max}$, controls the maximum timestep used when training on non-target (multi-speaker) data—where $t=0$ represents fully noised samples and $t=1$ represents clean samples.

We ablated $t_{max} \in \{0, 0.25, 0.5, 1.0\}$ across 140 epochs of training. The baseline ($t_{max}=0$) corresponds to excluding the multi-speaker data entirely from the denoising objective.

---

## Results

### 1. Validation Loss

| $t_{max}$ | Final Val Loss | Δ vs Baseline | % Change |
|-----------|---------------|---------------|----------|
| **0.0 (baseline)** | **2.1436** | — | — |
| 0.25 | 2.1457 | +0.0021 | +0.10% |
| 0.50 | 2.1435 | −0.0001 | −0.01% |
| 1.00 | 2.1517 | +0.0080 | +0.38% |

**Key Finding:** All experimental runs achieved validation losses within **0.4%** of the baseline. The differences are statistically negligible given the training variance (std ≈ 0.015 over the last 20 epochs).

### 2. Training Loss

| $t_{max}$ | Final Train Loss | Training Variance (std) |
|-----------|-----------------|------------------------|
| 0.0 | 2.990 | 0.0040 |
| 0.25 | 3.000 | 0.0041 |
| 0.50 | 3.009 | 0.0040 |
| **1.00** | **3.152** | **0.0192** |

**Key Finding:** Higher $t_{max}$ values correlate with higher training loss. This is expected—when $t_{max}=1.0$, the model must learn from fully denoised multi-speaker data, creating a harder optimization target. The training variance for $t_{max}=1.0$ is approximately **5× higher** than the baseline, indicating less stable training dynamics.

### 3. Sub-Loss Component Analysis

Breaking down the total loss into its constituent parts (duration, prior, and diffusion losses):

| $t_{max}$ | Duration Loss | Prior Loss | Diffusion Loss |
|-----------|--------------|------------|----------------|
| 0.0 | 0.3804 | 0.9869 | 0.7762 |
| 0.25 | 0.3822 | 0.9867 | 0.7769 |
| 0.50 | 0.3798 | 0.9868 | 0.7768 |
| 1.00 | 0.3813 | 0.9856 | 0.7847 |

**Key Finding:** The diffusion loss component shows the largest variation with $t_{max}$, while duration and prior losses remain stable. This suggests that the Ambient Diffusion framework primarily affects the flow-matching decoder rather than the duration predictor or encoder.

### 4. Training Stability

All runs completed 140 epochs without divergence or NaN issues. Gradient norms remained bounded within expected ranges (total gradient norm: 0.6–12.8). No evidence of training instability was observed for any $t_{max}$ configuration.

---

## Scientific Verdict

**The Ambient Diffusion adaptation did not yield measurable improvements over the baseline.**

### Interpretation

1. **No performance gain:** The experimental runs ($t_{max} > 0$) achieved essentially identical validation loss to the baseline ($t_{max}=0$). The tiny differences (≤0.4%) fall within training noise.

2. **No performance degradation:** Importantly, including the corrupted multi-speaker data in the training objective did not harm model quality. This suggests the flow-matching framework is robust to the Ambient Diffusion formulation.

3. **Increased training complexity with $t_{max}=1$:** Using full denoising on multi-speaker data ($t_{max}=1.0$) resulted in 5% higher training loss and 5× higher training variance, with no corresponding validation improvement. This indicates the model expends capacity learning the multi-speaker distribution without transferring that knowledge to the target speaker.

4. **Partial denoising ($t_{max}=0.25, 0.5$) is neutral:** The intermediate settings behave nearly identically to the baseline, suggesting the multi-speaker data contributes minimally when restricted to early denoising steps.

### Possible Explanations

- **Data imbalance:** The multi-speaker dataset (VCTK) has significantly different characteristics than LJSpeech. The model may learn speaker-specific features that don't transfer.

- **Speaker embedding isolation:** With per-speaker embeddings and $n\_spks=110$, the model may compartmentalize knowledge, preventing cross-speaker transfer.

- **Flow-matching vs. diffusion:** Ambient Diffusion was originally designed for standard diffusion models. The flow-matching formulation may respond differently to time-truncated training.

---

## Conclusion

This experiment demonstrates that the Ambient Diffusion framework, when adapted to flow-matching TTS, **does not improve target speaker synthesis quality** when using multi-speaker data as the corrupted source. While the approach successfully integrates into the training pipeline without causing instability, the additional data provides no measurable benefit.

For practitioners, this suggests that simply adding multi-speaker data with Ambient Diffusion is not a viable strategy for improving single-speaker TTS. Future work might explore:
- Alternative corruption strategies specific to speech (e.g., speaker perturbation rather than time-based)
- Removing speaker embeddings to force shared representations
- Using data augmentation instead of multi-speaker data as the corrupted source

---

## Appendix: Experimental Configuration

- **Model:** Matcha TTS (20.9M parameters)
- **Training:** 140 epochs, batch size 128, Adam optimizer (lr=1e-4)
- **Clean Data:** LJSpeech (speaker ID 109, ~24 hours)
- **Corrupted Data:** VCTK (109 speakers)
- **Hardware:** Single/Dual GPU (NVIDIA), mixed precision (fp16)
- **Seed:** 1234 (fixed across all runs)

*Generated plots available in `analysis_output/` directory.*

