# Kalman RNN (Orhan & Ma)

Domain language for the Orhan & Ma (2017) Kalman-filtering RNN experiment as reconstructed in this repo.

## Language

**Trial**:
One stimulus sequence of length `stim_dur`: population spikes over time, a Target trajectory, and an Opt mean trajectory.
_Avoid_: batch (a Batch contains many Trials); episode; sample

**Batch internals**:
Optional generative and filter quantities for a Batch — Gain, Expected rate, and Opt variance — returned only when requested for analysis or visualization, not used in training.
_Avoid_: diagnostics; Batch (the training fields alone)

**Batch**:
One minibatch of Trials: population spike inputs, Target, and Opt mean, each shaped `(batch, time, features)`.
_Avoid_: sample set; episode; Trial (unless a single sequence is meant)

**Target**:
The true latent stimulus `s` used as the supervised training signal.
_Avoid_: label when `opt_mean` is meant; posterior mean

**Gain**:
The hidden per-timestep variable `g` that scales every neuron’s tuning-curve rate and thereby controls the reliability of the population spikes. Orhan & Ma call this **input gain**; use that phrasing only as a paper-facing synonym for Gain, not for Population count.
_Avoid_: RNN input; Opt variance; population spike count; Input gain when Population count is meant

**Expected rate**:
The Poisson mean for a neuron at one timestep, determined by its preferred stimulus, the current Target, and Gain before a spike count is sampled.
_Avoid_: spike count; RNN input

**Population count**:
The total spike count across all input neurons at one timestep, which provides the RNN with observable evidence about response reliability.
_Avoid_: Gain; Input gain; Expected rate

**Prior precision**:
The filter’s precision about `s` immediately before incorporating the current timestep’s spikes — the uncertainty carried forward from history after the dynamics step.
_Avoid_: Opt precision when the post-update posterior is meant; Gain; Population count

**Effective Kalman gain**:
The weight the network places on current evidence when updating its estimate, a composite of this-step observation reliability (Population count / likelihood) and Prior precision. Operationally estimated as the coefficient relating the readout innovation update \(\hat{y}_t - (1-\gamma)\hat{y}_{t-1}\) to the innovation (Instantaneous population estimate minus that prediction). In this task Gain is independent across timesteps, so raw correlation of Effective Kalman gain with decoded Prior precision is the primary association (Population-count partialling is optional).
_Avoid_: Gain; Input gain; Opt precision; Decoder coefficient; raw \(\Delta\hat{y}\) without an innovation

**Instantaneous population estimate**:
The spike-count-weighted mean preferred stimulus at one timestep, summarizing the current population input without using temporal history. Its single-timestep likelihood standard deviation is `sqrt(sigtc_sq / Population count)`, so more spikes yield a tighter estimate.
_Avoid_: Opt mean; network readout; Target; Opt variance (which accumulates temporal history)

**Opt mean**:
The optimal Kalman filter’s posterior mean of `s`, used for evaluation against the network, not as the training target.
_Avoid_: target; ground truth

**Opt variance**:
The optimal Kalman filter’s posterior variance of `s`, expressing its remaining uncertainty after incorporating population spikes through the current timestep — the preferred uncertainty quantity for decoding and Concept-eraser analyses.
_Avoid_: gain; estimation error; variance of the Opt mean across Trials; Opt precision when the variance itself is meant

**Opt precision**:
The reciprocal of Opt variance (`1 / Opt variance`), the posterior precision *after* the current update — not the default decoding target in this project.
_Avoid_: Opt variance when the reciprocal is meant; Prior precision; Gain; estimation error

**Training loss**:
Final-timestep MSE between network readout and Target.
_Avoid_: fractional RMSE; full-sequence MSE (unless explicitly chosen)

**Fractional RMSE**:
How much worse the network’s RMSE is than the Opt mean’s RMSE, on final-timestep scalars — the paper’s progress metric, not the quantity optimized by Adam.
_Avoid_: training loss; plain RMSE when the comparison-to-optimal story is intended

**Mean activity**:
The across-unit mean of the hidden state at one Trial × timestep.
_Avoid_: Population count; Expected rate; network readout

**Kurtosis (sparsity)**:
The Fisher kurtosis of the hidden state across units at one Trial × timestep; higher values indicate a more peaked / sparse pattern than a Gaussian.
_Avoid_: Gain; Opt variance; Decoder R²

**Statistic correlation**:
An analysis that correlates a scalar hidden-layer summary (Mean activity or Kurtosis) with Opt variance across pooled Trial × timesteps, without fitting a Decoder.
_Avoid_: Decoder; linear decoding; LEACE eraser

**Decoder**:
A probe that maps pooled hidden states to a scalar Trial quantity (e.g. Opt mean or Opt variance) and is scored by held-out R².
_Avoid_: network readout (the trained RNN output); eraser; Statistic correlation

**Concept eraser**:
An edit to hidden states that removes linearly available information about a chosen concept, so that concept is linearly guarded in the edited representation.
_Avoid_: Decoder; finetuning; scrubbing of network weights

**One-step INLP**:
A Concept eraser that orthogonally projects hidden states onto the complement of a Decoder axis in ambient space (here the Opt-variance Decoder), zeroing that coordinate exactly. A refit linear Decoder can often recover the concept when \(\Sigma_{XX}\) is anisotropic.
_Avoid_: LEACE eraser; naive projection; multi-step INLP when only a single ambient nullstep is meant

**LEACE eraser**:
The least-squares Concept eraser of Belrose et al. (2023): the unique affine edit that linearly guards the concept while minimizing mean squared change to the hidden states.
_Avoid_: One-step INLP; orthogonal projection erasure; RLACE; multi-step INLP when the closed-form least-squares eraser is meant

**Per-timestep centering**:
At each timestep, subtract the across-Trial mean from hidden states and from every scalar Decoder/eraser target, with those means estimated on train Trials only and reused on validation/test.
_Avoid_: global pooling without removing the shared temporal trajectory; re-estimating means inside each evaluation split
