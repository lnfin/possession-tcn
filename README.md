# Broke Boys: ball possession from automatic tracking

More Than A Hack 2027. From the provided tracking of BAR-ATH, the code labels every video frame
`H`, `A`, `D` or `X`. The model is a temporal CNN followed by a three-state Viterbi decoder.
It was trained only on open matches from other sources, and every value with a real choice was
picked on those matches by a fixed procedure. The technical document is
[docs/technical_document.md](docs/technical_document.md).

## Run

Python 3.11 and four pinned packages:

```bash
python3.11 -m venv .venv-run
.venv-run/bin/pip install -r requirements.txt
```

Put the three provided files of the match in one folder, under `tactical/`:

```
$MTAH_DATA_DIR/tactical/BAR-ATH_tactical_atd.txt
$MTAH_DATA_DIR/tactical/BAR-ATH_tactical_atd.xml
$MTAH_DATA_DIR/tactical/BAR-ATH_tactical_smart_tagging.xml
```

The video and the homography are not read. Then one command writes the submission:

```bash
MTAH_DATA_DIR=/path/to/data .venv-run/bin/python run.py
```

The output is `out/submission.csv`, with 199,319 rows. Its md5 is
`9e98378fb243e8b8467ff71883acabb8`. We got the same md5 on macOS arm64 from a clean clone of this
repository, and on Linux x86_64 from a clean clone of the same code and weights at the freeze,
each time with only `requirements.txt` installed. The run takes under half a minute and
needs about 1.6 GB of RAM. It uses no GPU, no network and no randomness. The model weights are in
`mtah/weights/`: `seed0.npz`, `seed1.npz` and `seed2.npz`, plus `decode.json`.

## Layout

| path | what |
|---|---|
| `run.py` | the one command: provided files to CSV |
| `mtah/` | the inference path in numpy: loaders, ball cleaning, features, network forward pass, decoder; the open-data loaders, reference labels and generators used in training |
| `mtah/weights/` | the trained weights (3 seeds) and the decoder values |
| `training/` | the TCN in torch and the training script |
| `analysis/clean/` | selection: bank of degraded copies, metric harness, decoder plateau rule, gates, the pod steps |
| `tools/pipeline_check.py` | the single pipeline check on the public five minutes (prints PASS/FAIL only) |
| `tests/` | unit tests, static checks, an end-to-end flow on synthetic data |

## Rebuilding the weights

Training and selection ran on one GPU pod (RunPod, RTX 5090, about 80 minutes of compute) from
open data only. No file of the provided match was on that machine. The image was
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` (Python 3.12, torch 2.9.1+cu128).
`training/requirements.txt` is what the setup script installs on top of it, and
`training/pip_freeze_pod.txt` records the full environment of the run.

The setup script expects this repository at `/workspace/tree`:

```bash
git clone <this repository> /workspace/tree && cd /workspace/tree
bash analysis/clean/pod_setup.sh /workspace        # environment, Metrica, IDSSE download and parse
source /workspace/env.sh
for s in p2 p3 p3b p4 p4eval p5 p6 p8train p8 p9 p10bands registry; do
  python -m analysis.clean.pod $s --work /workspace/work --price <USD per hour> --spent <USD so far>
done
python -m analysis.clean.equiv --work /workspace/work --weights /workspace/work/weights
python -m analysis.clean.pod registry --work /workspace/work --price <USD per hour> --spent <USD so far>
```

The tests (`pytest tests`) need the training environment; the checks that read the open
matches or the organisers' repository are skipped when those are absent.

`p7` runs only when two or more candidates pass the gates. In our run none did. The bank of
degraded copies is deterministic in its seeds. GPU training is not bit-exact across devices, so a
rebuild gives weights close to ours but not identical.

## External data

- **Metrica Sports sample data**: three matches, tracking and events,
  <https://github.com/metrica-sports/sample-data>, commit `e706dd506b360d69d9d123d5b8026e7294b13996`.
- **IDSSE / DFL open data**: seven matches (two from the Bundesliga, five from the
  2. Bundesliga), tracking and events, CC BY 4.0, from figshare. Cite as: Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of
  spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195.
  <https://doi.org/10.1038/s41597-025-04505-y>

The code downloads both sources itself and does not redistribute them. Nobody hand-labelled
anything for this work. The training labels come from the providers' own event and per-frame
possession data.

## Licence

MIT, see [LICENSE](LICENSE). `mtah/third_party/games.py` is the organisers' file (their repository
is MIT-licensed), included unchanged. The match data is not in this repository and is not covered by
the licence.
