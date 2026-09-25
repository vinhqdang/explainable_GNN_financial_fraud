"""Executed on the Colab VMs: start the tuning queues (resumable; finished
configurations are skipped). The work is split between two VMs: VM "a" tunes
RTXGNN configurations 0-9 and five graph baselines, VM "b" RTXGNN
configurations 10-19 and the other three baselines. The final 10-seed runs of
RTXGNN are started separately once both halves are complete."""
import subprocess
import sys
vm = sys.argv[1] if len(sys.argv) > 1 else "a"
env = "RTX_DATA=/content/data RTX_DEVICE=cuda PYTHONPATH=.. MALLOC_MMAP_THRESHOLD_=65536"
G = {"a": "gas,sage,gcn,tgn,fraudre", "b": "gat,sefraud,tgat"}[vm]
R = {"a": "--cids 0-9", "b": "--cids 10-19 --tag _b"}[vm]
jobs = {
    # VM "a" also runs the final 10-seed RTXGNN runs once the records of VM "b" are complete
    "t1": f"{env} python run_tuning.py --models rtxgnn --n 20 {R} --threads 1"
          + (f"; test $(cat ../results/tuning_rtxgnn_b.jsonl | wc -l) -ge 30 && "
             f"{env} python run_tuning.py --models rtxgnn --final --threads 1" if vm == "a" else ""),
    "t2": f"{env} python run_tuning.py --models {G} --n 20 --threads 1; "
          f"{env} python run_tuning.py --models {G} --final --threads 1",
}
out = subprocess.run("ps aux | grep -c '[r]un_tuning'", shell=True, capture_output=True, text=True).stdout.strip()
if out != "0":
    print("jobs already running:", out)
else:
    for k, c in jobs.items():
        subprocess.Popen(f"cd /content/rtx/experiments && nohup sh -c '{c}' >> /content/{k}.log 2>&1 &",
                         shell=True, start_new_session=True)
    print("launched")
