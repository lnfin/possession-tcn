"""The money forecast of table 3.3 and the decision table of step P3b, section 4.3.

    t(config) = t_centre * prod_f (t_f / t_centre)
    forecast  = spent + (sum t(remaining configs of P3b) + N_left t(config)) / 3 * price
                      + sum T_CPU (32 / vCPU) * price
"""

FACTORS = {"steps": (1500, 3000, 6000), "channels": (32, 64, 128), "blocks": (7, 8, 9),
           "lr": (1e-3, 2e-3, 4e-3), "dropout": (0.0, 0.1, 0.2), "wd": (1e-5, 1e-4, 1e-3),
           "batch": (8, 16, 32), "chunk": (1500, 3000, 6000), "r0": (0.5, 0.75, 1.0)}
CENTRE = {"steps": 3000, "channels": 64, "blocks": 8, "lr": 2e-3, "dropout": 0.1, "wd": 1e-4,
          "batch": 16, "chunk": 3000, "r0": 0.75}
ORDER = ("steps", "channels", "blocks", "batch", "chunk", "lr", "dropout", "wd", "r0")
LARGEST = ("steps", "channels", "blocks")     # the largest plateau value within the limit
LIMIT_P3B = 6.65                              # the cut of table 3.3 at the ceiling of $7, section 4.3
N_LEFT = {"P4": 12, "P5": 36, "P7": 12, "P8": 4, "P9": 3}
T_CPU_H = {"P6": 40 / 60.0}                   # CPU steps of table 3.2 after the bank
PRIOR_MIN = 1.1 * 3                           # minutes per training before the first measure


def configs():
    """The 19 configurations: the centre, then one factor at a time."""
    out = [dict(CENTRE)]
    for f in ORDER:
        for v in FACTORS[f]:
            if v != CENTRE[f]:
                out.append(dict(CENTRE, **{f: v}))
    return out


def dropped_values(names, blocks, batch):
    """Values of the factors removed by table 3.3; chunk needs blocks, batch the first measure."""
    out = {}
    for f in names:
        if f in ("wd", "dropout"):
            out[f] = 0.0
        elif f == "chunk":
            rf = 2 ** blocks - 1
            out[f] = min(v for v in FACTORS["chunk"] if v >= 4 * rf)
        elif f == "batch":
            out[f] = batch
    return out


def time_of(chosen, t_centre, t_factor):
    """Minutes of a training with the chosen values; t_factor {(factor, value): minutes}."""
    t = t_centre
    for f, v in chosen.items():
        t *= t_factor.get((f, v), t_centre) / t_centre
    return t


def forecast(spent, chosen, t_centre, t_factor, n_left, remaining_p3b_min, price, vcpu, cpu_h=None):
    """Dollars of the whole graph with `chosen` for the remaining trainings."""
    cpu = T_CPU_H if cpu_h is None else cpu_h
    train_min = remaining_p3b_min + sum(n_left.values()) * time_of(chosen, t_centre, t_factor)
    return spent + train_min / 60.0 / 3.0 * price + sum(cpu.values()) * (32.0 / vcpu) * price


def choose(scores, t_centre, t_factor, spent, price, vcpu, plateau_eps, dropped):
    """Values of step P3b. scores {(factor, value): score}, the centre under each factor's centre value.

    dropped {factor: value}: factors removed by table 3.3, their values come from that table.
    Returns (values, decision table)."""
    n_left = dict(N_LEFT)
    chosen, table = dict(CENTRE), []
    for f in ORDER:
        if f in dropped:
            chosen[f] = dropped[f]
            table.append({"factor": f, "dropped": dropped[f]})
            continue
        vals = FACTORS[f]
        s = {v: scores[(f, v)] for v in vals}
        best = max(s.values())
        plat = [v for v in vals if s[v] >= best - plateau_eps]

        def cost(v):
            return forecast(spent, dict(chosen, **{f: v}), t_centre, t_factor, n_left, 0.0, price, vcpu)

        if f in LARGEST:
            order = sorted(plat, reverse=True)
        else:
            order = sorted(plat, key=lambda v: -s[v])
        fit = [v for v in order if cost(v) <= LIMIT_P3B]
        pick = fit[0] if fit else min(plat, key=cost)
        chosen[f] = pick
        table.append({"factor": f, "scores": {str(v): s[v] for v in vals}, "plateau": plat, "pick": pick,
                      "forecast": cost(pick), "over_limit": not fit})
    return chosen, table
