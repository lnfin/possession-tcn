# Broke Boys: technical document

More Than A Hack 2027. Ball possession on BAR-ATH from automatic tracking.

## 1. What is submitted, and what it replaces

The submission is `Broke Boys.csv`, md5 `9e98378fb243e8b8467ff71883acabb8`. It is the output of
`run.py` in this repository on three provided files: the ATD tracking, its XML metadata and the
Smart Tagging file.

**Our earlier system broke the rules and is not submitted.** It had rule-based modules, about 176
tuned values and several competing versions. Its thresholds, mechanisms and versions were chosen
with the help of human-made data about this match:

- hand labels on stretches of the match: the public five minutes, labelled by two team members,
  and windows w1 to w3, labelled by one of them. One member labelled the public window blind and
  then compared the labels with its official ground truth;
- per-frame human verdicts on individual frames;
- runs on the public five minutes against their ground truth, used for tuning and for choosing
  between versions.

The rules forbid hand labels on the provided game for any purpose, including validation and
choosing values, even when the labels never enter the pipeline. They also forbid using the public
five minutes for tuning. Some of those runs on the windows were made as late as 1 November 2026.

We did not try to repair that system value by value. We rebuilt it from a written protocol, frozen
before any of the new work, with md5 `536e88b900f8a67fb2e3f9fe8f61924f`. The protocol fixes four
things:

- which mechanisms exist;
- where every remaining value comes from;
- the decision procedure, run by script on open matches from other providers;
- the checks allowed on this match after the freeze.

The protocol and the run journal stay in our private records. This document states every rule it
relies on.

Parts of the old code base were carried over by a closed list: loaders, the feature builder, the
open-data generators, the reference builder and the numpy network. Every value chosen on this match
was removed from them. No old weight, value table or bank was reused. The old system is not used as
a candidate, a baseline or a comparison anywhere in the rebuild. Every decision in the rebuild has
its source recorded in the journal; memory and notes from the earlier work were not used as a
source.

## 2. Approach

```
ATD + XML -> drop ball spikes (> 40 m/s to both neighbours) -> 113 per-frame features
          -> temporal CNN, 3 seeds averaged -> P(H, A, D) per frame
Smart Tagging "Kick Off (Start)" -> start of play in each graded block; earlier frames are D
          -> Viterbi over H, A, D per half -> CSV; X outside the graded blocks
```

The system is one sequence model trained on degraded copies of open matches, plus one decoder. The
inference path has no hand-written possession rules, no whistle or audio, no video, no team-shape
model and no fitting of any kind on the match.

We chose this design because the earlier system's breach sat in its rule thresholds. Open matches
have no audio, no real Smart Tagging and not this tracker, so those thresholds cannot be re-chosen
honestly on other data. Removing the mechanisms together with their values was safer than
re-deriving them.

**Features (113 columns, team level, no player identity).** Coordinates are turned so that +x is
the home side's attacking direction in each half. The tracking itself gives that direction: it is
whichever way the home block faces, judged by which team's block lies behind the other's. The
columns are:

- ball presence, position, speed and distance outside the pitch lines;
- per team, the nearest and second nearest distances to the ball, and the counts within 3, 5 and
  10 m;
- control at three radii r0·(1, 2, 4), with r0 = 0.75 m. A team controls the ball at a radius if
  one of its players is within it and closer than any opponent. The columns hold the controlling
  team, the last and next controlling team, and the time since or until control;
- the time since or until the ball was last or next seen, and where it was then;
- team-block statistics over 1 s and 3 s windows: centre, spread, extent, speed, own-half share,
  and the gaps between the blocks;
- player speed.

**Network.** A non-causal temporal CNN at 25 fps:

- a 1×1 input layer to 128 channels;
- 9 residual blocks, each a kernel-3 dilated convolution (dilations 1 to 256, about ±20 s of
  context);
- a 1×1 output layer to three logits.

It is trained with cross-entropy on the reference R (section 3), using AdamW with a one-cycle
schedule (`pct_start` 0.1). The settings are 6000 steps, lr 2e-3, weight decay 1e-3, dropout 0.1,
batch 16 and chunks of 3000 frames. The block design (kernel 3, dilation 2^i) and `pct_start` were
inherited from earlier work and not searched. Three seeds are trained on all ten open matches, and
their probabilities are averaged.

Inference is a numpy re-implementation in float64. Probabilities are rounded before decoding, so
last-bit differences between machines are unlikely to reach the labels. The CSV was identical on
the two platforms we tried.

**Decoder.** A Viterbi pass over {H, A, D} in each half. A direct H↔A change costs 64 nats, and so
does a change between a team and D. There is no minimum spell length and no merging.

**Law 8.** This is the IFAB law on the start of play. Within each graded block, the frames before
the earliest `Kick Off (Start)` tag are D. That tag is the only Smart Tagging input to the model.
Outside the blocks published by the organisers, the label is X.

## 3. Data

**Provided match.** The model reads:

- the ATD `.txt`;
- from the `.xml`, the pitch size and the team of each slot;
- the Kick Off (Start) tags.

It does not use the video, the homography or any other tag. After the freeze, two checks in
section 5 read more: the possession-phase tags for a team-polarity check, and the attack direction
in the XML. The public ground truth was read once, by the pipeline check in section 5, after the
freeze.

**Open matches (all training and selection data).** Ten full matches. The three Metrica matches
last 94 to 97 minutes each, and the IDSSE matches are full league matches, so the total is roughly
16 hours of play.

- Metrica Sports sample data, matches 1 to 3, repository commit `e706dd5`;
- IDSSE / DFL, seven matches (CC BY 4.0; Bassek, Rein, Weber & Memmert, *Scientific Data* 12, 195,
  2025):
    - J03WMX and J03WN1, from the Bundesliga;
    - J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9, from the 2. Bundesliga, all five Fortuna
      Düsseldorf home matches.

J03WR9 is used for training but not for evaluation, because its goalkeeper track is mislabelled.

**Who labelled the training data.** We made no labels ourselves for the submitted system, on this
match or on any other. The earlier, replaced system is described in section 1. The labels come from
the data providers:

- Metrica's event stream for its three matches;
- the DFL per-frame `BallStatus` and `BallPossession` flags for the IDSSE matches.

The providers produced these themselves; event data of this kind is normally logged by their
operators. We only converted them into frame labels by script, using the rules below.

How the conversion works:

- **IDSSE:** the frame's flags give the owner and whether the ball is in play.
- **Metrica:** the owner is the team of the latest owning event: set piece, pass, carry, recovery,
  ball lost, ball out, shot or fault received. Challenges, cards and substitutions own nothing.

**Possession rules.** On Metrica, the rules below follow from the provider's event definitions.
IDSSE uses its own per-frame flags. The network learns these conventions from the references; the
decoder holds no possession rule of its own.

- **Loose ball.** It stays with the last team that owned it. A touch logged without an owning
  event, such as a challenge that does not win the ball, changes nothing. A block or a save is
  logged as a recovery, so it gives the defending team a short possession.
- **Change of owner.** It lands on the new team's first touch.
- **Dead time.** It runs from the stoppage to the restart:
    - for a ball out, the stoppage is the event's end, at most 3 frames earlier (`out_lag`);
    - for a goal, it is the shot's end;
    - for a foul, offside or handball, it is the event.

    When several events fall within 39 frames of the last one before the restart (`cluster`),
    the strongest reason sets the stoppage: goal, then ball out, then referee.
- **Restart.** After any restart there is always an owner.
- **Half edges.** Before the first event of a half and after its last, the ball is dead.

Issue #17 asked whether an uncontrolled, won header switches the owner. It was closed without an
answer, so that case is decided only by these references.

The earlier work had cross-checked Metrica's event types against the public ground truth. The
rebuilt reference builder follows only the provider's event definitions document and the Laws of
the Game.

**References used for scoring (E_def), with equal weights.**

| reference | definition |
|---|---|
| R | the raw reference above |
| R(25) | R with team spells shorter than 1 s merged into the previous team label |
| R(75) | the same with 3 s |
| R_def | a team spell is kept only if it contains a completed pass or shot of that team; otherwise it is merged |

A spell that follows D is never merged, so the D frames are identical in all four references. The
merge lengths and the R_def threshold (one event) are our choices, and the results depend on them.
The R_def threshold is a weaker version of the challenge text. None of these choices changed after
the protocol was frozen.

**Degraded copies.** We never train on clean tracking. A generator G turns each open match into
copies that look like an automatic tracker:

- **Players:** smoothing, position error, track breaks, missing tracks, and pieces assigned to the
  wrong team. Everything is laid out in 46 slots (24 + 22), a layout taken from this match's XML.
  The slot budget is asymmetric between teams, so pieces are lost when slots are full.
- **Wrong-team pieces:** these start at player contacts. Where contacts are too few for a copy's
  drawn share, the rest are placed uniformly, with no contact. That part of G is no different from
  the independent model T2 described below. The protocol asks us to report the realised share of
  uniform pieces, and we cannot: the per-copy parameter files stayed on the training machine,
  which was deleted. The bank is deterministic in its seeds, so the share can be regenerated.
- **Ball:**
    - smoothing, error and a lag of 0 to 5 frames;
    - visibility gaps, more likely when players crowd the ball, when it is high and when it is
      fast;
    - a single parameter calibrates the gap rate so that the ball is hidden on a share h of each
      copy's frames.
- **Ball out of play:** tails after the ball leaves play, and a ball carried to the restart.
- **Symmetry:** a team swap and a mirror in y, each with probability ½.

Most generator values are drawn per copy from a prior set by the kind of quantity, not by an old
value:

| kind of quantity | prior |
|---|---|
| share | U(0, 1) |
| duration | log-uniform 1 to 60 s |
| position error | log-uniform 0.05 to 1 m |
| spread | U(0, 2) |
| slope | U(0, 2) |

The exceptions:

- missing and wrong-team shares: U(0, 0.5);
- smoothing σ in G: log-uniform 0.5 to 16 frames for the ball, 0.5 to 8 frames for players;
- smoothing in T2: odd windows of 1 to 31 frames;
- ball lag: an integer from 0 to 5 frames;
- contact radius: log-uniform 0.5 to 3 m;
- crowd radius: U(1, 6) m;
- tail decay: U(0.2, 4) s;
- speed of a ball carried to the restart: U(1, 12) m/s;
- label coupling of gaps in T2: log-uniform 0.5 to 2;
- time scale of position noise: fixed at 12 frames in G, drawn as a duration in T2.

The only number from the provided match in the priors is the share of graded frames without a
ball, 41 %, published in the FAQ. The width of ±0.25 around it is our choice, which gives
h ~ U(0.16, 0.66).

The training bank has 6 copies of G per match. Selection uses two other families of copies:

- **T1:** G with priors 1.25× wider;
- **T2:** an independent corruption model. Its gap rate depends on the label, in either direction.

A third family, **T3**, puts T1 and T2 at the edges of their priors. It was sealed until the last
selection step.

**Assets tried earlier and dropped:**

- the video and any detections from it;
- the audio whistle;
- hand-written possession, dead-ball and set-piece rules;
- a team-shape model;
- dead-ball evidence from Smart Tagging set pieces;
- hazard rates measured on this match's ATD;
- a false-out noise and a stuck-ball corruption, both from studying this match;
- every hand label on this match.

## 4. How values were chosen

Thirteen values had a real choice:

- 9 training values: steps, channels, blocks, lr, dropout, weight decay, batch, chunk, r0;
- 2 decoder costs;
- 2 reference-builder values: `out_lag` = 3 and `cluster` = 39 frames.

The two reference-builder values were computed first on a local machine, from open-match events
only. They were recomputed on the selection machine with an equality check. Everything else was
selected by script on that machine: a cloud GPU that never held any file of the provided match.

The remaining constants were set before any run, either by rule (physics, the data format, the
published data notes) or by the authors' judgement. The protocol lists each one with its source.
The frozen values are recorded in a registry with md5 `38aaa0d6695342f2ff02475d8ea4745a`.

**Units and summary.** There are six evaluation units:

- Metrica 1, Metrica 2 and Metrica 3;
- J03WMX;
- J03WN1;
- the four Fortuna Düsseldorf home matches, as one unit.

The summary W gives Metrica and IDSSE half the weight each. A unit's score is the official
`spellF1` against the four references E_def, averaged over its T1 and T2 copies. The official
scorer is called directly.

**Noise.** ε_s is the noise band of W. It is twice the weighted spread of the score difference
between two training seeds of the base model, with a floor of 0.019 set before the run. ε_m is the
same for macroF1, with a floor of 0.005.

**Candidates.** The protocol fixed exactly three, each tested against the base model S0. Their
numbers are protocol identifiers; no other candidate existed.

| candidate | what it adds |
|---|---|
| C1 | an auxiliary head that predicts possession boundaries, extra loss weight near changes, and an H↔A cost that drops where a boundary is likely |
| C2 | five columns on the ball at rest and on kicks, aimed at restarts |
| C7 | the same inputs and network, trained on R(75) instead of R |

**Training values.** We varied one factor at a time on three folds (metrica1, J03WMX, J03WN1):

| factor | levels |
|---|---|
| steps | 1500, 3000, 6000 |
| channels | 32, 64, 128 |
| blocks | 7, 8, 9 |
| lr | 1e-3, 2e-3, 4e-3 |
| dropout | 0, 0.1, 0.2 |
| weight decay | 1e-5, 1e-4, 1e-3 |
| batch | 8, 16, 32 |
| chunk | 1500, 3000, 6000 frames |
| r0 | 0.5, 0.75, 1 m |

The plateau of a factor is every level scoring within 0.025 of its best. Each pick must keep the
cost forecast of the whole run under a limit:

- steps, channels and blocks take the largest plateau value that fits;
- the other factors take the best-scoring plateau value that fits;
- if no plateau value fits, the cheapest is taken.

The factors were not searched jointly. The three fold matches also count later in the evaluation.

**Decoder.** We search a grid of the two costs, keeping only cells with switch ≤ 2·dead. A cell's
*shifted cells* are the cells with both costs multiplied by ½ or by 2, which is what a factor-2
change in network confidence would amount to. The *regret* of a cell is its score after a shift,
on one reference, minus the best score on that reference; it is never positive. The rule works on
out-of-fold predictions:

- keep the cells within ε_s of the best that also lie on the plateaus of each source and of the T2
  copies alone;
- among them, keep those whose worst regret over shifts ½, 1, 2 and over the four references is
  within ε_s of the best worst regret r*;
- take the cell deepest inside that set.

The value that ships is chosen on all six units. To score a unit, the same rule is re-run without
it (nested). Before the nested runs, the grid was extended upwards to 144 using all matches, so
that extension is not nested.

**Gate G0.** The base model must beat a nearest-player baseline on the same copies by 0.04 on W, on
at least 5 of 6 units and on both sources. The baseline follows the logic of the organisers'
starter code, with its smoothing window chosen the same way.

**Candidate acceptance.** A candidate is accepted only if all of the following hold. Here T is its
gain in W over S0.

1. T ≥ ε_s.
2. A one-sided sign-flip test over the six units gives p ≤ 1/64, with Holm's stopping rule across
   the three candidates.
3. It does not lose on either source by more than that source's noise, and its worst reference is
   no worse than S0's by more than ε_s.
4. It does not lose on the T2 copies alone.
5. For C2 only, it gains at least the IDSSE noise on IDSSE. On Metrica copies, the ball at rest in
   dead time is written by our generator, so only IDSSE can show a real gain there.
6. It loses no more than ε_m on frame-level macroF1.
7. Conditions 1 and 3 still hold without each unit in turn, in at least 5 of 6 folds.

A candidate with 0 ≤ T < ε_s could still pass on the tie-break metric macroF1, by ε_m and with the
same test. None did.

**Outcome.** These scores are against automatic references on open matches. They are not an
estimate of the score on BAR-ATH.

| step | result |
|---|---|
| training values | steps 6000, channels 128, blocks 9, lr 2e-3, dropout 0.1, wd 1e-3, batch 16, chunk 3000, r0 0.75. Every level of every factor was on its plateau, and every pick fit the limit |
| decoder | switch 64, dead 64. All six nested choices gave the same cell; r* = −0.067. The plateau along the diagonal is 2 cells wide. Only four cells keep both shifted cells inside the plateau (switch 48, dead 24 to 64); the chosen cell is not one of them |
| noise | ε_s = 0.019, ε_m = 0.005, both at their floors |
| base model S0 | W = 0.611. W(S0 − baseline) = 0.470: higher than the baseline on 6 of 6 units, 0.493 on Metrica, 0.447 on IDSSE. G0 passes |
| C1 | T = 0.009 < ε_s, p = 1/64: not accepted. Its boundary weight was set to a neutral 1.0, replacing an earlier 2.0 chosen near the windows |
| C2 | T = 0.004, p = 0.16, and it failed condition 5: not accepted |
| C7 | T = −0.024, worse on all units: not accepted |
| transfer between sources | trained on IDSSE, scored on Metrica: 0.609 (baseline 0.165). Trained on Metrica, scored on IDSSE: 0.504 (baseline 0.117). The same decoder cell came out in both directions |
| sealed stress bank T3 | S0 0.441, baseline 0.081 |

The final model is S0, trained on all ten matches with three seeds.

## 5. Freeze and checks on the match

The freeze was recorded in our journal before the first run of `run.py` on BAR-ATH. The record
holds:

- the md5 of the registry of values, of every weight file and of the protocol;
- the code tree hash;
- the isolation record of the selection machine.

Before the freeze only one check had touched BAR-ATH. It checked file reading and units, and
printed PASS/FAIL only.

After the freeze, all of the following passed:

- the official validator, on the exact file;
- consistency checks on the full match without labels, which print PASS or FAIL only:
    - X exactly outside the blocks;
    - D before kick-off;
    - home/away polarity against the team field of the possession-phase tags;
    - frame alignment against ball-out events;
    - invariance to a 180° rotation (100 %), to a team swap and to a mirror, against thresholds
      set on open-match copies;
    - attack direction against the XML;
    - spell-length and dead-share bands from the open matches;
    - agreement between seeds;
    - feature coverage;
    - tag reading;
- identical output from two runs on macOS arm64 and from a clean clone of this repository on
  macOS;
- identical output on Linux x86_64, from a clean clone of the same code and weights at the
  freeze. In the rebuild, this run is the only time the match files were on a cloud machine; the
  selection machine never held them.

The public five minutes were used once, after the freeze, by `tools/pipeline_check.py`. It runs
five pass/fail checks through the official metric functions:

- format;
- frame alignment within ±25 frames;
- H/A not swapped;
- D meaningful;
- no degenerate output.

All five passed. The script prints no number, and nothing was changed afterwards.

## 6. Deviations from the protocol's procedure

**The forecast limit for the training values changed the selected values.** The protocol set the
limit on the cost forecast of every training-value pick (section 4) at $2.85. That number belonged
to an earlier budget. The protocol raised the same limit to $5.7 for its stop-and-cut rules but
missed this place.

Before the search started, money already spent, much of it on idle machines, made $2.85
unreachable. When the budget ceiling was raised to $7, we set this limit to $6.65, the cut
threshold of the raised budget. The change was recorded before the training-value search started,
without seeing any training result.

With $2.85, the forecast was already above the limit at the centre of the grid ($2.98). We
recomputed the rule from the recorded scores and times: no plateau value of any factor would have
fit, so it would have taken the cheapest value of every factor. That gives steps 1500, channels
32, blocks 7, lr 4e-3, dropout 0, weight decay 1e-5, batch 8, chunk 1500 and r0 0.5. With $6.65
the rule took the values in section 4. This change of procedure changed all nine training values,
and we disclose it as such.

The following deviations did not change any selected value:

- **Cloud budget.** The team raised the ceiling from $6 to $7. It also dropped the requirement of
  at least 24 vCPUs while the forecast stayed under the ceiling; the selection machine had 20.4.
  The spend was about $3.1, of which about $1.45 went on two idle machines. They sat idle for 71
  and 41 minutes, beyond the protocol's 10-minute limit, before they were deleted. Nothing ran on
  them except a setup script that failed in its first minute (next item), and they produced no
  data or results.
- **Environment.** The setup script installs packages with `--break-system-packages`, because of
  PEP 668 on the image.
- **Two readings of the checks.**
    - The feature-coverage check prints only its verdict, not the list of out-of-range columns,
      because the checks print PASS/FAIL only.
    - The swap and mirror thresholds, "min − 25 %", are read as 0.75 × the minimum, as for the
      bands.
- **Five planned departures from our own compliance checklist:**
    - no held-out subset of open matches beyond the leave-one-unit-out folds, since only six units
      exist; the transfer between sources plays that role, weakly;
    - the input-reading check on BAR-ATH before the freeze, described in section 5;
    - some constants set by the authors' judgement, with disclosure;
    - a fallback baseline window computed on a local machine near the match files, from open
      matches only;
    - the Linux reproduction run after the freeze.

## 7. Limitations and residual traces

**What the method cannot show.**

- **References are not the ground truth.** Gains on Metrica and IDSSE may not carry over, and no
  number in this document estimates the score on BAR-ATH.
- **The generator is not this tracker.** Wide priors buy robustness, not accuracy. The network may
  be less confident on real input than on the copies, which can fragment possessions, and the
  decoder costs in nats are tied to that confidence.
- **The decoder cell is not robust to a factor-2 shift.** The shipped cell (64, 64) does not keep
  both of its shifted cells inside the plateau, so a factor-2 change in confidence on BAR-ATH can
  move it off the plateau. The worst regret r* = −0.067 bounds the loss on open-match copies
  only. The spell-length bands passed on BAR-ATH, which rules out only gross fragmentation.
- **Dead ball while the ball is visible.** The base model may see it poorly, because the inference
  path has no rule channels and no whistle.
- **Few units.** There are six units, five of the IDSSE matches come from one club, and the folds
  share data. So the real size of the sign-flip test is above its nominal value, and the nested
  estimate is slightly optimistic. A weak but real improvement can be rejected.
- **Training-value search.** It used three matches that later count in the evaluation, one factor
  at a time. Its centre was inherited from earlier work on this match.

**Cases outside the training domain.**

- **Kick-off frames.** IDSSE starts a half at kick-off; Metrica has dead frames before its first
  event. The network saw pre-kick-off frames only on Metrica.
- **Removed corruptions.** A stuck, motionless ball and a goalkeeper assigned to the wrong team
  were removed from the generator, because both came from studying this match. On BAR-ATH such
  cases are outside the training domain.
- **Dead ball on Metrica copies.** Our generator writes it, because the clean ball vanishes there.
  On IDSSE the dead ball is real.

**What we knew about the match when designing.**

- The features and the forms of degradation were designed by people who had looked at this ATD,
  which the rules allow.
- The generator priors were written by the same people. Their centres were not built from the
  match, but the authors knew it.
- The earlier contact and crowd radii (1 m and 3 m) came from this ATD. They were replaced by the
  physical ranges 0.5 to 3 m and 1 to 6 m, which contain them.
- The ball-lag mechanism was measured on this match's video. Its range of 0 to 0.2 s is set by
  rule.
- The candidate set and the base architecture were chosen by people who had seen the windows. The
  decisions between them were taken by script on open matches.
- The control-radius grid includes earlier values, and its centre comes from earlier work. The pick
  was made on open matches.

**Other choices of ours.**

- We use the published graded blocks as segment boundaries, which the rules allow.
- A tag a few frames after the start of the first block yields a few D frames in live play.

**What we would do with more time.**

- Get real Smart Tagging on open matches, to test set-piece dead-ball evidence.
- Hand-label open matches under the challenge's definition, to replace the merge thresholds.
- Add more independent sources, so that small gains could be accepted or rejected with power.

## 8. Reproduction

```bash
python3.11 -m venv .venv-run && .venv-run/bin/pip install -r requirements.txt
MTAH_DATA_DIR=/path/to/data .venv-run/bin/python run.py      # out/submission.csv
```

`MTAH_DATA_DIR/tactical/` holds the three provided files. The output md5 is
`9e98378fb243e8b8467ff71883acabb8` on macOS arm64 and on Linux x86_64. The weights ship with the
repository, and the README describes how to retrain them from open data. The bank of copies is
deterministic in its seeds; GPU training is close to the original but not bit-exact across devices.

## 9. External data and licences

- Metrica Sports sample data: <https://github.com/metrica-sports/sample-data>, commit
  `e706dd506b360d69d9d123d5b8026e7294b13996`.
- IDSSE / DFL open data, CC BY 4.0: Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An
  integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195.
  <https://doi.org/10.1038/s41597-025-04505-y>

Our code is MIT-licensed. Two pieces are the organisers' MIT-licensed code, included under their
licence (`mtah/third_party/LICENSE`): `mtah/third_party/games.py`, unchanged, and the `smooth`
function in `analysis/clean/baseline.py`. The repository contains no match data, no ground truth
and no per-frame output.
