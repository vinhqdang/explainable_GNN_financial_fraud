"""Per-time-step class counts of Elliptic (Appendix table)."""
import os
from common import ROOT
from rtxgnn.data import load_elliptic, describe

d = load_elliptic()
df = describe(d)
split = lambda t: "train" if t <= 30 else ("val" if t <= 34 else "test")
rows = []
half = (len(df) + 1) // 2
L = df.iloc[:half].reset_index(drop=True); R = df.iloc[half:].reset_index(drop=True)
fmt = lambda r: f"{int(r.step)} & {split(r.step)} & {int(r.licit):,} & {int(r.illicit):,} & {int(r.unknown):,}"
for i in range(half):
    right = fmt(R.iloc[i]) if i < len(R) else "& & & &"
    rows.append(fmt(L.iloc[i]) + " & " + right + r" \\")
tot = df[["licit", "illicit", "unknown"]].sum()
out = os.path.join(ROOT, "manuscript", "revision1", "tables", "elliptic_steps.tex")
with open(out, "w") as f:
    f.write("\\begin{tabular}{rlrrr|rlrrr}\n\\toprule\n")
    f.write("Step & Split & Licit & Illicit & Unlab. & Step & Split & Licit & Illicit & Unlab.\\\\\n\\midrule\n")
    f.write("\n".join(rows) + "\n\\midrule\n")
    f.write(f"\\multicolumn{{2}}{{l}}{{Total}} & {int(tot.licit):,} & {int(tot.illicit):,} & {int(tot.unknown):,} & \\multicolumn{{5}}{{l}}{{}}\\\\\n\\bottomrule\n\\end{{tabular}}\n")
print(open(out).read()[:500])
