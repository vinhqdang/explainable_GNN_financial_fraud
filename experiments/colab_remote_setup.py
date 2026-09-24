"""Executed on a fresh Colab VM by colab_resume.sh: clone/update the code,
install dependencies and prepare Elliptic."""
import subprocess
cmds = [
    "cd /content && (test -d rtx || git clone -q https://github.com/vinhqdang/explainable_GNN_financial_fraud.git rtx) && cd rtx && "
    "mkdir -p /content/res_bak && cp results/*.jsonl /content/res_bak/ 2>/dev/null; git fetch -q origin && git reset -q --hard origin/main && "
    "cp /content/res_bak/*.jsonl results/ 2>/dev/null; git log --oneline | head -1",
    "pip install -q torch_geometric xgboost",
    "mkdir -p /content/rtx/results && cd /content/rtx && RTX_DATA=/content/data PYTHONPATH=. python -c 'from rtxgnn.data import load_elliptic; load_elliptic()'",
]
for c in cmds:
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(c[:60], "->", r.returncode, r.stderr[-300:] if r.returncode else "")
