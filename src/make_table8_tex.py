"""Emit the LaTeX form of Table 8 (Theorem 2.4 terms) from the measured CSV."""
import csv, os
from pathlib import Path

from paths import RESULTS_ROOT
TAB = Path(os.path.expanduser(RESULTS_ROOT + "/map_project/paper/tables"))
rows = list(csv.DictReader(open(TAB / "table8_theorem_terms.csv")))
d = {}
for r in rows:
    d.setdefault(r["dataset"], {})[float(r["sigma"])] = r
SIG = [0.05, 0.10, 0.20]
lines = [r"\begin{table}[!t]", r"\centering",
  r"\caption{Theorem~2.4 decomposition, measured. All terms are per-image MSE "
  r"($\times 10^{-4}$) under the deployed hard-routing policy, for which the "
  r"diversity term is identically zero. \emph{Matched optimal} is the fraction of "
  r"images for which the $\sigma$-matched agent is also the per-image optimal agent, "
  r"and explains the collapse of routing regret. The final column gives the "
  r"corresponding condition under uniform routing, where diversity is active.}",
  r"\label{tab:theorem}",
  r"\begin{tabular}{llrrrrr}", r"\toprule",
  r"Dataset & $\sigma$ & Spec.\ gain & Routing regret & Condition & Matched optimal & "
  r"Cond.\ (uniform) \\", r"\midrule"]
for ds in ["BraTS 2021", "nodule3d", "CBSD68"]:
    for i, s in enumerate(SIG):
        r = d[ds][s]
        cond = float(r["condition_hard_mse"]) * 1e4
        cond_s = (r"\textbf{%+.2f}" % cond) if cond > 0 else ("%+.2f" % cond)
        lines.append("%s & %g & %.2f & %.2f & %s & %.1f\%% & %+.2f \\\\" % (
            ds if i == 0 else "", s,
            float(r["specialization_gain_mse"]) * 1e4,
            float(r["routing_regret_hard_mse"]) * 1e4,
            cond_s,
            100 * float(r["oracle_picks_matching_agent_frac"]),
            float(r["condition_uniform_mse"]) * 1e4))
    if ds != "CBSD68":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "table8_theorem_terms.tex").write_text("\n".join(lines) + "\n")
print("wrote", TAB / "table8_theorem_terms.tex")
