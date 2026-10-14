# Proposed instrument repair v1 — approval required; NOT implemented

The v2 diagnostic found stable physical-dose latents but inadequate existing
MLP instruments (agent differential RMS8.98/10.25/15.96px; natural absolute
agent57.25px, block14.91px, angle.501rad on independent families). A constant
source/render bias does not explain the discrepancy. This does **not** prove
that LeWM lacks agent-position information or that a new probe will pass.

## Proposed bounded next task

1. Explicitly approve the prospective domain before fitting: retain unchanged
   Step2 fresh-start goal/render/preprocessing, or approve a separately versioned
   alternative. Do not choose goals/codecs on held-out readout performance.
2. Keep encoder/predictor frozen and preserve all original MLP files/results.
   New instruments get a new artifact/version path, architecture/seed/target
   transforms, complete training-family lineage, hashes and a selection manifest.
   Never overwrite `.stable-wm/probes/pusht_*_mlp.pt`.
3. Train only on **natural unpatched observations** with independently verified
   simulator labels. No predicted/patched latents, Ridge projector outcomes,
   contact-effect H2 results or future confirmatory families enter fitting or
   model selection. Existing exposed32 diagnostic families may be explicitly
   assigned to development only; none can validate the revised instrument.
4. Proposed feasibility allocation:80 natural training families,24 disjoint
   model-selection/calibration-validation families,60 fresh final measurement
   validation families. These are resource targets, not a power guarantee.
   Related poses/doses/trajectories/appearances share the highest family ID.
   Freeze geometry, exclusions, budgets and analysis before final validation;
   no outcome replacement or post-test sample expansion. Bootstrap families,
   and seek precision through prespecified design, not pseudo-replicated frames.
5. Start with the existing small256-hidden-unit MLP architecture for positions,
   fixed target normalization fitted on training only and a small predeclared
   learning/regularization schedule selected on validation only. A separately
   versioned sin/cos angle head is a proposal requiring approval, not an implicit
   change to the old scalar readout; evaluate wrapped-radian error and reject
   degenerate near-zero predicted angle vectors without arbitrary normalization.
   Do not search unlimited architectures until an appealing H2 result appears.
6. Include natural agent doses0/10/20/30px across positions/orientations/free and
   near-contact; also independent block-position doses1/2/5/10px and angle
   changes.01/.025/.05/.10rad, where geometry is valid. This validates non-target
   instruments and the plan's1-px downstream effect scale, not just large
   agent movements. Use coherent natural dynamics as an additional readout
   calibration population; record exact full physical labels and actions.
7. Retain fixed candidate5-px/.10-rad absolute/large-dose caps. Calibrate proposed
   differential margins on development/validation, require both calibrated and
   practical limits on fresh final families, and do not increase caps for PASS.
   For1-px block-effect claims, require an independently validated differential
   error budget <=.5px, with a one-sided95% upper bound; this is the proposed
   half-effect measurement allocation, not an empirically established tolerance.
   It may fail even if broader position tests pass. Report dose-specific errors,
   zero-change leakage, bias, direction/magnitude and family intervals.
8. Reassess natural teacher-forced/open-loop forecasts with both original and
   new instruments. Natural-state accuracy does not validate arbitrary predicted
   or patched OOD latents. Retain latent forecast errors, natural encoded future
   controls and uncertainty; label physical readout conclusions conditional on
   the instrument. No new predictor training to rescue the result.

If this protocol fails, report measurement-limited/INCONCLUSIVE and propose the
next change for review. Passing readout calibration only reopens the observation
gate for human review; it does not establish H1, selectivity, H2 or authorize H3.
**No fitting or formal Step3 collection is authorized by this proposal.**
