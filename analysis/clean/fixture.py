"""The fixture of run.py: one G copy of J03WMX (seed 0, two periods of 7500 frames) as provided files.

    python -m analysis.clean.fixture --bank BANK --out DIR

Writes DIR/tactical/ in the formats of the match: ATD .txt, EPTS .xml, SportsCode .xml with a
Kick Off (Start) tag per block, and DIR/blocks.json. An open match; the files stay out of the repo.
"""

import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from analysis.clean import bank, metric
from mtah import paths
from mtah.external import corrupt

PERIOD_FRAMES = 7500
GAP = 500
LEAD = 25
NAME = "J03WMX"


def _meta_xml(slot_team, pitch, right_first):
    players = "".join('<Player id="P{}" teamId="T{}"><Name>Track {}</Name></Player>'.format(i, k, i + 1)
                      for i, k in enumerate(slot_team))
    refs = "".join('<SplitRegister separator=","><PlayerChannelRef playerChannelId="playerP{0}_x"/>'
                   '<PlayerChannelRef playerChannelId="playerP{0}_y"/>'
                   '<PlayerChannelRef playerChannelId="playerP{0}_s"/></SplitRegister>'.format(i)
                   for i in range(len(slot_team)))
    direction = "left_to_right" if right_first else "right_to_left"
    teams = ('<Team id="T0"><Name>Home</Name><ProviderTeamsParameters><ProviderParameter>'
             '<Name>attack_direction_first_half</Name><Value>{}</Value></ProviderParameter>'
             '</ProviderTeamsParameters></Team><Team id="T1"><Name>Away</Name></Team>').format(direction)
    return ('<main><Metadata><Sessions><Session id="1"><FieldSize><Width>{:g}</Width><Height>{:g}</Height>'
            '</FieldSize></Session></Sessions><Teams>{}</Teams><Players>{}</Players></Metadata>'
            '<DataFormatSpecifications><DataFormatSpecification><SplitRegister separator=";">{}'
            '</SplitRegister></DataFormatSpecification></DataFormatSpecifications></main>'
            ).format(pitch[0], pitch[1], teams, players, refs)


def _tags_xml(tags):
    rows = []
    for i, (code, start, kind) in enumerate(tags):
        rows.append('<instance><ID>{}</ID><code>{}</code><start>{:.3f}</start><end>{:.3f}</end>'
                    '<label><group>Type</group><text>{}</text></label></instance>'.format(
                        i, escape(code), start, start + 2.0, escape(kind)))
    return "<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format("".join(rows))


def _num(v):
    return "NaN" if not np.isfinite(v) else "{:.5f}".format(v)


def write(bank_root, out):
    ref = bank.load_refs(bank_root, NAME)
    game = metric.load_game(NAME)
    Pm, team, Bm, _, period, _, _ = bank.make_copy(NAME, 1, 0, game, ref["Y"], ref["log"])
    slot_team = corrupt.SLOT_TEAM
    n_slots = corrupt.N_SLOTS
    L, W = game["pitch_m"]
    tracts = [np.flatnonzero(period == h)[1000:1000 + PERIOD_FRAMES] for h in (1, 2)]
    players = np.full((PERIOD_FRAMES * 2 + GAP, n_slots, 2), np.nan)
    ball = np.full((PERIOD_FRAMES * 2 + GAP, 2), np.nan)
    players[:PERIOD_FRAMES] = Pm[tracts[0]] / np.array([L, W])
    ball[:PERIOD_FRAMES] = Bm[tracts[0]] / np.array([L, W])
    players[PERIOD_FRAMES + GAP:] = Pm[tracts[1]] / np.array([L, W])
    ball[PERIOD_FRAMES + GAP:] = Bm[tracts[1]] / np.array([L, W])
    blocks = [[1, 0, PERIOD_FRAMES - 1], [2, PERIOD_FRAMES + GAP, 2 * PERIOD_FRAMES + GAP - 1]]
    folder = Path(out) / "tactical"
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    speed = np.zeros(n_slots)
    for t in range(len(ball)):
        cols = ";".join("{},{},{}".format(_num(players[t, s, 0]), _num(players[t, s, 1]),
                                          _num(speed[s])) for s in range(n_slots))
        lines.append("{}:{}:{},{},NaN".format(t + 1, cols, _num(ball[t, 0]), _num(ball[t, 1])))
    stem = "{}_tactical".format(paths.GAME)
    (folder / (stem + "_atd.txt")).write_text("\n".join(lines) + "\n")
    (folder / (stem + "_atd.xml")).write_text(_meta_xml(slot_team, (L, W), True))
    tags = [("SET PIECES", (b[1] + LEAD * k) / 25.0, "Kick Off (Start)") for k, b in enumerate(blocks)]
    tags.append(("BUILD UP", 100.0, "none"))
    (folder / (stem + "_smart_tagging.xml")).write_text(_tags_xml(tags))
    (Path(out) / "blocks.json").write_text(json.dumps(blocks))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(write(a.bank, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
