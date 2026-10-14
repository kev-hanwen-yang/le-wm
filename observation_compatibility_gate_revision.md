# Proposed observation gate revision — review required, not activated

## Separate decisions

1. **Exact historical provenance:** UNRESOLVED. Continue reporting that status;
   it is not, by itself, a permanent barrier to an independently validated
   prospective observation contract.
2. **Compatibility-qualified Step 3 development pilot using the current
   observation/readout combination:** NOT SUPPORTED by this assessment. The
   predeclared nuisance/practical checks failed, including twice the proposed
   pilot drift budgets. This is a measured limitation, not missing-records logic.
   A separately scoped observation/readout-calibration exercise is defensible as
   diagnostic development, with the incompatibility explicitly retained; it must
   not be relabeled a passing Step 3 pilot or interpreted as causal evidence.
3. **Freeze for subsequent confirmatory testing:** NOT SUPPORTED. No contract
   was frozen and `encoder_dataset_ready` remains false.

These are recommendations for review. Neither this document nor a successful
audit process exit authorizes collection. No full Step 3 dataset was generated.

## What the independent compatibility route should require

- Keep the existing Step 1 model/preprocessing/action/time contracts and Step 2
  fresh-start mechanics gate as mandatory, unchanged prerequisites.
- Declare the **actual prospective production** observation pipeline, including
  marker position/visibility, task-goal semantics, geometry/colors, camera size,
  raw resize and saved-pixel format. The prior comparison candidate has the
  central marker `(256,256,pi/4)`; current mechanics fixtures use a safe physical
  goal `(350,120,0)` and a correspondingly rendered marker. This assessment does
  not quietly select one or alter goals to reconcile them. A comparison candidate
  is not a frozen production contract, and `_goal`/goal encoding remain unchanged.
- Establish matching at the resolution relevant to the observation. For static
  RGB comparisons, independently verified render-relevant geometry is sufficient
  in principle: hidden dynamics that do not enter the renderer need not become
  an impossible historical prerequisite. Do not call this full-state restoration.
  For history/dynamics claims, use full current-backend readbacks and reproducible
  trajectories, not public-state-only matching of historical moving frames.
- Calibrate natural same-state nuisance and independent readout measurement
  resolution on development families. Check independent validation families and
  planned orientation/workspace/contact/history strata before freeze. A few
  attractive images, low pixel RMSE or a nearer latent neighbor cannot pass.
- Use practical error budgets tied to intended physical contrasts, alongside
  empirical natural-nuisance envelopes. The current proposals allocate at most
  1.25 world px per agent/block observation and 0.025 rad per angle observation
  to pilot-level drift; they derive from one quarter of the plan's preservation
  budgets, not a pixel threshold. For a 1-px informative block-effect contrast,
  tighten block nuisance to 0.25 px per observation, and independently validate
  the readout's ability to resolve that effect. These are proposed allocations
  requiring review; they cannot be enlarged simply because data fail them.
- Report paired latent L2 changes relative to natural same-state variation and
  matched natural motion at intended 10/20/30-px scales. Preserve missing/zero
  denominators. No arbitrary fixed latent norm or pixel-RMSE cutoff is sufficient.
- Freeze prospective pipeline, units, reference families, margins, exclusions
  and decision procedure before further independent validation/confirmatory
  outcomes. Changing a pipeline or readout creates a versioned development
  protocol and needs fresh validation; protected tests stay untouched.

**Avoid a circular gate:** fitting Ridge and measuring its actual injected norm
are later Steps 5–6. They are not prerequisites for the *observation-contract*
freeze needed to collect the data used by those steps. Use natural physical-dose
benchmarks now; when an actual projector exists, additionally verify nuisance
against its realized patch norm and independently read-out physical effects
before interpreting selectivity/H2. This clarifies the earlier assessment
protocol's discussion of actual patch scales; it changes no measured threshold
or result into a PASS.

## Precisely what remains necessary

1. Diagnose the substantial input/readout shift using development evidence:
   spatial edge/rasterization versus color/chroma/decoder effects. The fixed codec
   grid demonstrates sensitivity but does not reproduce the residual uniquely.
   Do not adopt the codec with the best validation number as the production fix.
2. Resolve **prospective**, rather than unknowable historical, renderer/goal
   semantics and evaluate the pipeline actually proposed for collection while
   preserving physics, task goals and official goal encoding. No change is
   implemented by this proposal.
3. Validate frozen independent physical readouts on that prospective domain.
   Legacy MLPs have incomplete family provenance, and their drift/absolute error
   is not currently small enough. If a separately versioned readout refit is
   proposed later, train it only on natural development labels, leave LeWM and
   original probes untouched, and test with independent families before causal
   work. Refitting an instrument does not automatically establish compatibility.
4. Obtain more diverse, unexposed validation states and stronger dose references.
   Here only five previously pixel-audited validation families and three
   development 10-px references were available. No test-driven resampling or
   final-test threshold selection is permitted.
5. Show nuisance/error bounds meet the frozen practical and empirical criteria,
   including effect-scale measurement resolution. Failure remains FAIL or
   measurement-limited, regardless of whether historical records are ever found.

This route can ultimately remove historical record availability as a blocker;
the present independent evidence simply does not yet pass it. Recovering the
original MP4 settings is helpful, **not the only allowed way forward**.

Methodological basis: [Lakens (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/)
motivates prespecified meaningful equivalence bounds, rather than treating lack
of a significant difference as equivalence. Our five-family bootstrap bounds are
descriptive feasibility checks, not a formal TOST or population-equivalence proof.
The [FFmpeg codec documentation](https://ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb)
supports explicit codec/pixel-format controls; it does not identify the historical
codec or justify choosing a favorable replacement after inspecting outcomes.
