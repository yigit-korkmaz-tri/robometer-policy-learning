# Flow-MILE: learning from intervention and nonintervention

This document captures the project's research intent and its implementation as inspected on
2026-10-23. It is intended as context for future development and experiment sessions. Hypotheses,
implemented behavior, and possible changes are identified separately; the hypotheses below are
not claims of established experimental results. Configurations can change, so inspect the saved
run config when interpreting an experiment.

## Research objective

Flow-MILE is our proposed approach for improving a pretrained flow-matching robot policy using
human-in-the-loop (HITL) interaction. Its central goal is to **reduce or eliminate dependence on
pretraining demonstration replay during subsequent learning**.

Our HG-DAgger baseline learns from human corrections, optionally aggregated with initial
demonstrations. Flow-MILE additionally learns from timesteps at which the human lets the robot
continue. These decisions provide implicit information about acceptable behavior, even though
the human supplies no action at those timesteps.

The intended mechanism is:

- Corrections teach the policy how to improve at problematic states.
- Nonintervention feedback supplies gradients at states where the human accepted robot behavior.
- Together, those signals should help preserve competence while learning from corrections, using
  the trajectories collected during correction rounds.

Starting from pretrained weights is part of the setting. The target is reduced **subsequent data
access**, not learning the entire task from scratch. Fixed warmup sets of 10, 30, or 50 demos are
useful limited-data conditions; a zero-demonstration-replay condition directly tests the central
goal.

## Code map

| Component | Source |
| --- | --- |
| Proposed algorithm and losses | [modeling_flow_mile.py](../robometer_policy_learning/algorithms/flow_mile/modeling_flow_mile.py) |
| Algorithm settings | [flow_mile.yaml](../robometer_policy_learning/configs/algorithm/flow_mile.yaml) |
| Flow actor and Euler sampling | [modeling_flow.py](../robometer_policy_learning/algorithms/flow_matching/modeling_flow.py) |
| Iterative and precollected HITL training | [train_hitl.py](../scripts/train_hitl.py) |
| HITL configuration | [robomimic_hitl.yaml](../robometer_policy_learning/configs/robomimic_hitl.yaml) |
| Collection and precollected-data loading | [hitl_utils.py](../robometer_policy_learning/utils/hitl_utils.py) |
| Recent-round replay | [round_replay_buffer.py](../robometer_policy_learning/buffers/round_replay_buffer.py) |
| Chunk sampling and label alignment | [samplers.py](../robometer_policy_learning/buffers/samplers.py) |
| Single-phase supervised training | [train_supervised.py](../scripts/train_supervised.py) |

## Policies, actions, and labels

Let $s$ denote a policy observation, including images and normalized proprioception when used.
Let $a$ denote an action chunk of horizon $H$, not necessarily one simulator control action.

- $\pi_\theta$: the trainable policy, implemented as `online_actor`.
- $\pi_{\mathrm{ref}}$: the currently frozen rollout/reference policy, implemented as
  `rollout_policy`.
- $\pi_k$: the policy that collected data in interaction round $k$.
- $a_h$: an observed human correction.
- $a_r$: a logged robot action or action chunk from a nonintervention period.

When EMA is enabled, `actor` is the deployment/evaluation copy and `online_actor` receives
optimizer updates. The HITL loop snapshots the deployment actor as the reference before each
training phase. One current reference model is held; the algorithm does not retain a separate
reference network for every historical round.

The dataset labels have distinct roles:

| `intervention` | Meaning | Direct imitation loss | Intervention BCE |
| --- | --- | --- | --- |
| `0` | Robot acts; human does not intervene | No | Yes, target 0 |
| `1` | Human correction | Yes | Yes, target 1 |
| `2` | Optional offline/warmup demonstration | Yes | No |

Nonintervention is treated as implicit acceptance under the human's intervention rule. It does
not establish that the executed action is globally optimal. In particular, the current method
does not simply behavior-clone every accepted robot action.

## Flow-matching action scores

For an action chunk $a$, draw $x_0\sim\mathcal N(0,I)$ and $t\sim U(0,1)$. The implementation
uses the interpolant and target velocity

$$
x_t = [1-(1-\sigma_{\min})t]x_0 + ta,
\qquad
u = a-(1-\sigma_{\min})x_0.
$$

Define the conditional flow-matching loss

$$
\ell_\theta(a,s)
=\mathbb E_{t,x_0}\left[\|v_\theta(x_t,t,s)-u\|^2\right].
$$

The code averages squared error over the action-chunk dimensions and estimates this expectation
with `score_monte_carlo_samples` draws. Lower loss means a better flow-matching fit to that action.

Two score definitions are supported:

$$
r_\theta(a,s)=-\ell_\theta(a,s)
$$

or, with `reference_relative_score=true`,

$$
r_\theta(a,s)
=\ell_{\mathrm{ref}}(a,s)-\ell_\theta(a,s).
$$

The relative score measures improvement in fit against the frozen reference. These are
**likelihood proxies**, not exact evaluations of $\log\pi_\theta(a\mid s)$.

The connection to [Diffusion-DPO](https://arxiv.org/abs/2311.12908) is preference learning through
generative-model loss comparisons relative to a reference. Our objective uses a probit
intervention model and includes states with no observed human alternative. Exact equivalence
to DPO or to maximum likelihood under the actual flow-policy density is not assumed.

## Intervention likelihood

Write $\Phi$ for the standard normal CDF, $\beta$ for `probit_scale`, $c$ for
`intervention_cost`, and $w$ for `expected_rollout_score_weight`.

Initially, the robot baseline for a state is

$$
b_\theta(s)=\frac1K\sum_{j=1}^{K}r_\theta(a^0_j,s),
$$

where the $a^0_j$ are drawn from the frozen reference or a stored collection-policy pool.
The default baseline weight is $w=1$.

### Intervention: the human action is observed

With `condition_intervention_on_action=true`, a human-correction row uses

$$
p_\theta(\nu=1\mid s,a_h)
=\Phi\left(\beta[r_\theta(a_h,s)-w b_\theta(s)]-c\right).
$$

The BCE target is 1. This encourages a score comparison consistent with the human preferring
their correction over the collection-policy alternatives. The baseline generally consists of
sampled alternatives, not a recorded robot action paired with every human correction.

### Nonintervention: the human action is latent

For nonintervention rows, candidate human-like alternatives are sampled from the trainable
policy. The marginal is estimated as

$$
p_\theta(\nu=1\mid s)
\approx\frac1K\sum_{j=1}^{K}
\Phi\left(\beta[r_\theta(a^\theta_j,s)-w b_\theta(s)]-c\right),
\qquad a^\theta_j\sim\pi_\theta(\cdot\mid s).
$$

With `condition_nonintervention_on_robot=true`, the baseline on label-0 rows is replaced by

$$
b_\theta(s)=r_\theta(a_r,s).
$$

The BCE target is 0. The comparison asks whether candidate alternatives would justify
intervening over the robot action the human actually accepted. This is the mechanism that
extracts policy gradients from non-corrected timesteps.

If `condition_intervention_on_action=false`, intervention rows also use the marginal rather
than the observed-action probability. If `condition_nonintervention_on_robot=false`, label-0
rows retain the sampled reference baseline.

The displayed equations assume `normalize_score_gaps=false`. When enabled, running-statistics
normalization and optional clipping transform the score gap before the CDF. Probabilities are
also clamped to $[10^{-4},1-10^{-4}]$ before BCE.

## Combined training objective

The implementation minimizes

$$
\mathcal L
=\mathcal L_{\mathrm{FM,corrections/demos}}
+\lambda_I\mathcal L_{\mathrm{intervention}}
+\lambda_A\mathcal L_{\mathrm{anchor}}
+\lambda_P\mathcal L_{\mathrm{proximal}}.
$$

- **Imitation:** ordinary flow-matching loss on labels 1 and 2 only, averaged over that subset.
- **Intervention likelihood:** BCE on labels 0 and 1 only, averaged over that subset.
- **Anchor:** match the trainable and frozen reference velocity fields at noisy interpolants of
  reference-policy action samples, using the batch's states. This regularizes policy drift
  without requiring demonstration actions at those states.
- **Proximal:** optional score-based regularization implemented through `proximal_score_loss`;
  the experiments discussed here use weight 0.

The weights are additive coefficients. `lambda_intervention` does not multiply a complementary
`1-lambda_intervention` factor on the imitation loss. This objective does not require an
environment-reward optimization term.

The anchor regularizer and offline-demo replay are separate mechanisms. Keeping a pretrained
reference network or sampling its actions at HITL states does not itself require replaying the
original demonstrations.

## Stored rollout samples and their interaction with conditioning

The configuration emphasized in our discussions is:

```text
offline_algorithm.condition_intervention_on_action=true
offline_algorithm.condition_nonintervention_on_robot=true
offline_algorithm.use_stored_rollout_samples=true
```

For each new online transition, `precompute_rollout_samples` generates a pool of action chunks
from that round's frozen collection policy. Pools are attached to transitions and retained as
the policy changes. The current settings distinguish several different counts:

| Setting | Value in the inspected config | Meaning |
| --- | --- | --- |
| `stored_rollout_pool_size` | 50 | Chunks stored per online transition |
| `monte_carlo_samples` | 10 | Chunks subsampled from a stored pool per row/update; also the trainable-policy marginal sample count |
| `mc_num_inference_steps` | 10 | Euler integration steps used to generate a sampled chunk |
| `score_monte_carlo_samples` | 1 | Flow-matching noise/time draws per scored chunk |
| `anchor_monte_carlo_samples` | 1 | Fresh reference chunks used for anchor regularization per state |

Stored chunks are subsampled without replacement when the pool is large enough. Rows without
a pool use fresh reference samples. In the Can policy used in our example commands, each chunk
contains 8 control actions and the policy executes 6 before replanning. Those horizons come
from the loaded policy/run config and are separate from the counts above.

With both conditioning flags enabled, stored pools primarily affect the **intervention-row
baseline**: label-0 baselines are replaced with logged robot actions. The implementation still
draws fresh reference samples for the anchor and other score calculations.

Crucially, a stored pool freezes **actions**, not scores or reference-network weights. Historical
pool actions are rescored during each update. With reference-relative scoring, the reference
loss uses the current frozen `rollout_policy`; historical reference networks are not recovered
from the pool. Thus preserving collection-round sample distributions is a narrower guarantee
than preserving an entire historical preference model.

## Iterative learning hypothesis

The intended online procedure is:

1. Start from a pretrained flow policy, retaining its architecture. Our current policy examples
   use a transformer velocity network with an image encoder for image-based tasks.
2. Collect human-supervised rollouts. For Flow-MILE, keep both autonomous and correction steps
   with `hitl.store_only_human=false`.
3. Aggregate the selected trajectories with previous correction-round data. Optionally mix a
   fixed warmup dataset for limited-demo comparisons.
4. Freeze a copy of the deployment policy before the training phase. Populate sample pools on
   transitions that do not already have them.
5. Update the online policy with imitation, intervention likelihood, and configured regularizers;
   update its EMA deployment copy when enabled.
6. Evaluate autonomous task performance, then collect another round using the updated policy.

Our hypothesis is that repeated correction rounds gradually improve behavior that was initially
suboptimal but acceptable. Improved policies change which states are visited, and humans may
begin correcting finer errors as performance improves. Nonintervention feedback is intended to
preserve enough accepted behavior for that improvement to continue without demonstration replay.

This depends on informative feedback and coverage: if the human always accepts two behaviors,
the intervention labels alone cannot establish a preference between them. Old acceptance labels
can also become less representative when the human's standards change. These are hypotheses to
study, not guarantees of the objective.

Current replay controls include `hitl.recent_round_ratio` (0.5 in the inspected config), which
allocates half the online batch to the newest round once earlier rounds exist. The remaining
online samples come from older rounds. This is separate from `hitl.offline_sample_ratio`, which
controls mixing an offline anchor with the online buffer. `epoch_sampling` and `use_epochs`
control sampling passes and the training budget; record them when comparing experiments.

Current online collection keeps successful rollouts; `keep_only_hitl_rollouts=true` additionally
requires a human intervention. Consequently, label-0 data under that setting consists of
autonomous portions of selected corrected successes. It does not include every unattended
success or failure encountered during collection. Record this selection rule when discussing
what the nonintervention data represents.

## Gradient implementation: current behavior and an open design question

**Differentiable online sampling has been discussed but is not implemented in the inspected
code.** The current method receives gradients through action-score evaluation, correction
imitation, and enabled regularizers. Candidate actions themselves are treated as constants
during backward:

- `FlowMILE._sample_policy_actions` wraps sampling in `torch.no_grad()`.
- `FlowMatchingActor.sample_actions` and `sample_actions_batch` are also decorated with
  `@torch.no_grad()`.
- Reference-loss computation in `_action_scores` and `_score_stacked_actions` uses `no_grad()`.

For a marginal written as

$$
p_\theta(s)=\mathbb E_\epsilon[f_\theta(g_\theta(\epsilon,s),s)],
$$

where $g_\theta$ generates an action, a full pathwise derivative includes

$$
\nabla_\theta p_\theta
=\mathbb E_\epsilon\left[
\partial_\theta f_\theta
+\partial_a f_\theta\,\partial_\theta g_\theta
\right].
$$

Current sampling omits the second path. This is a surrogate-gradient choice, not evidence that
the whole intervention loss is detached, nor an established explanation of a performance
problem. With our conditioning flags, the missing trainable-sampling path concerns the
nonintervention marginal; corrected-action comparisons use fixed observed human actions and
fixed reference samples.

A flow-matching Euler sampler is differentiable in principle:

$$
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k,s).
$$

A possible alternative estimator would differentiate through these steps for online-policy
samples. Removing only the outer `no_grad()` block would not suffice because the actor sampler
has its own decorator. It would also increase memory use; action clipping introduces regions
with zero action gradients.

For reference-relative scores, frozen reference **parameters** must be distinguished from
reference **input gradients**. If $a_\theta$ is generated by the trainable policy,

$$
\nabla_\theta\ell_{\mathrm{ref}}(a_\theta,s)
=\partial_a\ell_{\mathrm{ref}}(a_\theta,s)
\,\partial_\theta a_\theta.
$$

Freezing reference parameters with `requires_grad_(False)` allows this input gradient.
Wrapping reference scoring in `no_grad()` blocks it. A full pathwise variant would preserve
action-input gradients through reference scoring while leaving its parameters frozen.
Logged human/robot actions and frozen-policy samples have no such trainable-sampler path.

Changing these gradient paths is a proposed ablation, not an automatically justified fix.
Do not describe the existing implementation as computing the full gradient of the stated
trainable-policy marginal.

## Data access and experimental conditions

Keep the following conditions distinct:

| Condition | Fine-tuning data |
| --- | --- |
| Zero demonstration replay | Only correction-round trajectories, including nonintervention steps |
| Limited warmup replay | Correction-round trajectories plus a fixed small demonstration subset |
| Pretraining replay | Correction-round trajectories plus the original pretraining demonstrations |

In `train_hitl.py`, `hitl.offline_mode=null` selects the first training-buffer condition.
`warmup` loads `hitl.warmup_dataset_path`; `pretraining` mixes the source demonstration dataset.
`self` creates an anchor from successful autonomous rollouts and is another experimental
condition, rather than external demonstration replay.

**Zero replay is not currently identical to zero pretraining-dataset access.** The inspected
HITL script still reconstructs environments from the source HDF5 and reads it for observation
normalization when `normalize_lowdim_obs=true`, even with `offline_mode=null`. A strict
no-access implementation needs to recover environment/scene information and normalization
statistics from saved artifacts instead. Preserve the pretrained policy's input normalization;
silently disabling it changes the inference contract.

## Iterative and precollected training entry points

`scripts/train_hitl.py` supports both online correction rounds and a single training phase on
a precollected HITL HDF5. For precollected training, use:

```text
hitl.precollected_hitl_dataset=data/YOUR_HITL_DATASET.hdf5
hitl.num_iterations=1
hitl.rollouts_per_iter=0
hitl.train_steps_per_iter=25000
hitl.use_epochs=false
hitl.human_mode=simulated
teleop.enable_render=false
```

This skips new intervention collection but still performs configured autonomous evaluations.
In this mode `human_mode=simulated` avoids the Enter prompt retained by `real` mode; it does not
load an expert or generate simulated interventions. Dataset `/meta` provenance must match the
configured `load_dir` and `checkpoint`. A precollected Flow-MILE run can populate stored pools
from that matching initial policy before its single training phase.

For Flow-MILE select `algorithm@offline_algorithm=flow_mile`,
`alg.offline_alg_name=flow_mile`, and `hitl.store_only_human=false`.
For our flow-based HG-DAgger baseline select `flow` for both algorithm settings and
`hitl.store_only_human=true`; precollected loading filters to contiguous human-correction
segments when no reweighting is configured.

`scripts/train_supervised.py` also supports single-phase Flow-MILE fine-tuning from a pretrained
flow actor, provided its dataset has both intervention classes. It uses `training.load_dir`
and optional `training.checkpoint`, rather than the top-level HITL fields. It does not provide
the HITL script's explicit two-buffer warmup mixing, and disables stored rollout pools on its
HDF5-buffer path. Our precollected HITL comparisons therefore use `train_hitl.py`.

With chunked policies, inspect labels and usable segment lengths. The chunk sampler aligns
the intervention label to its conditioning observation at the first timestep. Enabling
`hitl.segment_by_intervention` prevents mixed-label chunks, but segments shorter than the
action horizon cannot produce a full chunk. Keeping this treatment matched across comparisons
matters when attributing improvements to nonintervention learning.

## Existing Can experiment context

These paths identify the concrete setup discussed across sessions; they are not requirements
of the method:

| Artifact | Path or selection |
| --- | --- |
| Initial run | `checkpoints/can_image/2026-10-19_13-45-22` |
| Initial checkpoint | `checkpoint=suboptimal` (a mutable alias; inspect its `training_state.json` for the step) |
| Original demonstrations | `data/robomimic_datasets/robomimic_image/can/ph/image_converted.hdf5` |
| Fixed demo subsets | `data/warmup_can_10.hdf5`, `data/warmup_can_30.hdf5`, `data/warmup_can_50.hdf5` |
| Precollected HITL example | `data/can_hitl_rollouts_flow_10_demos.hdf5` |

The warmup subsets were sampled with seed 0 and are nested: 10 demos are contained in the
30-demo set, which is contained in the 50-demo set. They are subsets of the original
demonstrations, not human-correction datasets. The HITL example contains both intervention
classes, with provenance pointing to the initial run and `suboptimal` checkpoint above. Recheck
the file's `/meta` and the checkpoint's actual weights if either artifact is replaced; a reused
checkpoint name alone does not establish that collection and training weights still match.

Earlier command examples used learning rate `5e-5`, batch size 256, intervention weight 0.1,
anchor weight 0.1, and proximal weight 0. These were experiment choices. The inspected algorithm
YAML now has learning rate `1e-5`, batch size 64, intervention weight 0.1, anchor weight 0.01,
and proximal weight 0. A saved run's resolved config is authoritative.

## Evidence needed for the research claim

The central comparison is whether adding nonintervention feedback improves autonomous policy
performance and retention at a fixed correction-data budget, especially with zero demonstration
replay. Useful controlled comparisons include:

- Corrections-only flow imitation, with and without the same anchor regularizer.
- The same Flow-MILE objective with label-0 BCE terms removed, retaining the same data, anchor,
  correction loss, and sampling rules, to isolate the nonintervention gradient contribution.
- Imitation of corrections plus accepted robot actions, to test whether the preference model
  adds value beyond self-imitation.
- Zero replay, matched 10/30/50-demo warmup sets, and full demonstration replay.
- Iterative collection and fixed precollected-data training, reported as different settings.
- Stored versus fresh reference samples, conditioning flags, and sampler-gradient estimators.

Measure autonomous success across multiple seeds, human intervention burden, and task-specific
behavior quality over rounds. Also inspect correction fit, per-class intervention probabilities,
score margins, and retention. Lower intervention BCE or wider score margins alone do not
establish better behavior. Match initial weights, data availability, budgets, evaluation resets,
normalization, and regularizers when identifying the source of an improvement.

Nonintervention can be noisy because of reaction delays, attention, and changing tolerance.
Flow-loss scores can improve preference margins without improving generated actions. Both
limitations motivate the controlled experiments above. The iterative refinement hypothesis
and the claim that nonintervention gradients reduce replay dependence remain empirical.

For the broader intervention-learning motivation, see
[MILE: Model-based Intervention Learning](https://liralab.usc.edu/mile/).
