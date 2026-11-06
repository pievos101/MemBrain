# Exploratory fixed-tail control

Added after seeing preliminary results from the shape study. This is exploratory,
not a replacement for its frozen primary candidate or a held-out confirmation.
The source/protocol are separately hashed BEFORE any fixed-tail scores are
evaluated. There is no coefficient selection, attack fitting, or score-sign flip.

Observation: true prefix + true continuation is always the same original token
sequence. A causal target run on that sequence yields the same hidden states
and teacher probabilities at every prefix partition. A mean of the remaining
suffix changes because the selected token set changes. Centered true-suffix
trajectories therefore include moving-window composition, not only memory.

Use the same12 trained shape-study targets and candidate texts. Generate the
whole remaining suffix at each prefix as before, but pool/score ONLY the final
four token positions for both origins. Four is fixed because at least five
suffix tokens remain at the largest prefix. No target/score tuning is performed.
True-tail embeddings and teacher-forced loss should be exactly invariant over
prefixes; check this numerically for ALL candidates, not chosen examples.

Diagnostic scores: early-weighted area of true/generated tail cosine; negative
area of free-running CE against TRUE tail tokens; and negative true-tail teacher
loss. All are ranking scores, not calibrated probabilities. No membership labels
enter extraction/scoring. Labels enter only retrospective ROC/AP metrics.

Report all three scores and the invariance check. This control changes the
evaluation target to four tokens; its AUC must not be presented as a directly
matched whole-suffix accuracy comparison. No post-hoc promotion as best Membrain.
It addresses a measurement confound and does not prove trajectories contain no
membership information. Run after the primary benchmark:
`python experiments/membrain_tail_control.py`.
