#!/bin/bash
# Step P1 on the pod: environment, open matches, isolation record. No network volume, no match data.
#   bash pod_setup.sh /workspace
set -euo pipefail
W=${1:-/workspace}
cd "$W"
mkdir -p "$W/log"
python --version | tee "$W/log/python.txt"
# the image marks its Python as externally managed (PEP 668); the container is thrown away with the pod
pip install -q --break-system-packages --ignore-installed -r "$W/tree/training/requirements.txt" 2>&1 | tail -2
pip freeze > "$W/log/pip_freeze.txt"
python -c "import numpy, pandas, scipy, kloppy, torch; print(numpy.__version__, pandas.__version__, scipy.__version__, kloppy.__version__, torch.__version__, torch.cuda.is_available())" | tee "$W/log/versions.txt"

# the official scorer, one file, by commit
mkdir -p "$W/official/evaluation"
curl -sL -o "$W/official/evaluation/possession_metrics.py" \
  https://raw.githubusercontent.com/metrica-sports/more-than-a-hack-2027/f8f2629/evaluation/possession_metrics.py
echo "7718cb8a237452b48b35713984d31c75  $W/official/evaluation/possession_metrics.py" | md5sum -c -

# Metrica
git clone -q https://github.com/metrica-sports/sample-data "$W/ms"
git -C "$W/ms" checkout -q e706dd5
git -C "$W/ms" log -1 --format=%H | tee "$W/log/metrica_commit.txt"

cat > "$W/env.sh" <<EOF
export MTAH_DATA_DIR=$W/data
export MTAH_CACHE_DIR=$W/cache
export MTAH_EXTERNAL_CACHE=$W/cache/external
export MTAH_EXTERNAL_DIR=$W/ms/data
export MTAH_OFFICIAL_REPO=$W/official
export PYTHONPATH=$W/tree
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EOF
source "$W/env.sh"
mkdir -p "$MTAH_DATA_DIR" "$MTAH_CACHE_DIR"

# isolation record, section 3.2
{
  echo "== cpu.max"; cat /sys/fs/cgroup/cpu.max || true
  echo "== nvidia-smi"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
  echo "== df"; df -h || true
  echo "== mount"; mount || true
  echo "== find"
  find / \( -path /proc -o -path /sys -o -path /dev -o -path '*/site-packages' -o -path '*/dist-packages' \) -prune -o \
    \( -name '*BAR*' -o -name '*possession_gt*' -o -name '*atd_hazards*' -o -name 'weights*' -o -name '*.npy' \) -print || true
  echo "== du"; du -sh "$W"/* 2>/dev/null || true
} > "$W/log/isolation.txt" 2>&1

# IDSSE: download and parse, one process per match
cd "$W/tree"
cat > "$W/log/idsse.py" <<'EOF'
import sys
from mtah.sota import idsse
from mtah.external import idsse_open
m = sys.argv[1]
idsse.load(m)
idsse_open.ball_z(m)
print(m, "ok", flush=True)
EOF
pids=()
for m in J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9; do
  python "$W/log/idsse.py" "$m" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
du -sh "$MTAH_CACHE_DIR"
echo "P1 done"
