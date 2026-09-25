"""Executed on the Colab VM: start the tuning queues (resumable; finished
configurations are skipped). Chain 1 tunes RTXGNN, chain 2 the graph baselines;
each chain then re-runs its selected configurations with 10 seeds."""
import subprocess
env = "RTX_DATA=/content/data RTX_DEVICE=cuda PYTHONPATH=.. MALLOC_MMAP_THRESHOLD_=65536"
G = "gas,sage,gcn,tgn,fraudre,tgat,gat,sefraud"
jobs = {
    "t1": f"{env} python run_tuning.py --models rtxgnn --n 20 --threads 1; "
          f"{env} python run_tuning.py --models rtxgnn --final --threads 1",
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
