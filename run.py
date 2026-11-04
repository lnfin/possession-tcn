"""One command from the provided match files to the submission CSV.

    python run.py                                    # out/submission.csv

--blocks and --weights exist for the test fixture only: the blocks otherwise come from the official
games.py, the weights from mtah/weights/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out/submission.csv")
    ap.add_argument("--blocks", help="fixture only: JSON [[half, first, last], ...]")
    ap.add_argument("--weights", help="fixture only: folder with seed*.npz and decode.json")
    a = ap.parse_args()
    from mtah import infer, paths
    from mtah.transfer import ours
    blocks = [tuple(b) for b in json.loads(a.blocks)] if a.blocks else ours.blocks()
    labels = infer.run(paths.data_dir(), blocks, a.weights or infer.WEIGHTS_DIR)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, s) for i, s in enumerate(np.asarray(labels).tolist()))
    print("wrote", out, "rows", len(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
