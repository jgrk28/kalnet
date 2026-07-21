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
The hidden per-timestep variable `g` that scales every neuron’s tuning-curve rate and thereby controls the reliability of the population spikes.
_Avoid_: RNN input; Opt variance; population spike count

**Expected rate**:
The Poisson mean for a neuron at one timestep, determined by its preferred stimulus, the current Target, and Gain before a spike count is sampled.
_Avoid_: spike count; RNN input

**Population count**:
The total spike count across all input neurons at one timestep, which provides the RNN with observable evidence about response reliability.
_Avoid_: Gain; Expected rate

**Instantaneous population estimate**:
The spike-count-weighted mean preferred stimulus at one timestep, summarizing the current population input without using temporal history. Its single-timestep likelihood standard deviation is `sqrt(sigtc_sq / Population count)`, so more spikes yield a tighter estimate.
_Avoid_: Opt mean; network readout; Target; Opt variance (which accumulates temporal history)

**Opt mean**:
The optimal Kalman filter’s posterior mean of `s`, used for evaluation against the network, not as the training target.
_Avoid_: target; ground truth

**Opt variance**:
The optimal Kalman filter’s posterior variance of `s`, expressing its remaining uncertainty after incorporating population spikes through the current timestep.
_Avoid_: gain; estimation error; variance of the Opt mean across Trials; Opt precision

**Opt precision**:
The reciprocal of Opt variance (`1 / Opt variance`), the preferred uncertainty quantity for decoding analyses.
_Avoid_: Opt variance when the reciprocal is meant; Gain; estimation error

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
A probe that maps pooled hidden states to a scalar Trial quantity (e.g. Opt mean or Opt precision) and is scored by held-out R².
_Avoid_: network readout (the trained RNN output); eraser; Statistic correlation

**Concept eraser**:
An edit to hidden states that removes linearly available information about a chosen concept, so that concept is linearly guarded in the edited representation.
_Avoid_: Decoder; finetuning; scrubbing of network weights

**LEACE eraser**:
The least-squares Concept eraser of Belrose et al. (2023): the unique affine edit that linearly guards the concept while minimizing mean squared change to the hidden states.
_Avoid_: orthogonal projection erasure; RLACE; INLP when the closed-form least-squares eraser is meant

**Per-timestep centering**:
At each timestep, subtract the across-Trial mean from hidden states and from every scalar Decoder/eraser target, with those means estimated on train Trials only and reused on validation/test.
_Avoid_: global pooling without removing the shared temporal trajectory; re-estimating means inside each evaluation split
