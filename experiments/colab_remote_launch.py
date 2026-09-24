"""Executed on the Colab VM by colab_resume.sh: (re)start the GPU job queues.
All scripts skip runs whose records already exist, so relaunching resumes.
Two chains only: the VM has 2 vCPUs and 12 GB of RAM."""
import subprocess
env = "RTX_DATA=/content/data RTX_DEVICE=cuda PYTHONPATH=.. MALLOC_MMAP_THRESHOLD_=65536"
M = "lr,rf,xgb,mlp,gcn,sage,gat,tgat,tgn,apan,caregnn,pcgnn,gas,fraudre,sefraud,rtxgnn"
V = ("rtxgnn:full,rtxgnn:no_sparsity,rtxgnn:time2vec_abs,rtxgnn:no_suf,rtxgnn:no_nec,rtxgnn:no_curriculum,"
     "rtxgnn:focal_loss,rtxgnn:sparsity_x4,rtxgnn:no_sparsity_drop0,rtxgnn:no_sparsity_drop5,"
     "rtxgnn:no_sparsity_wd1e3,rtxgnn:dim32,rtxgnn:dim128,rtxgnn:layers1")
jobs = {
    "c1": f"{env} python run_elliptic.py --models {M} --seeds 2-4 --tag main_gpu --skip elliptic_main.jsonl --threads 1; "
          f"{env} python run_elliptic.py --models evolvegcn --seeds 0-9 --tag evo --threads 1; "
          f"{env} python run_elliptic.py --models mlp,gcn,sage,gat,sefraud,rtxgnn --seeds 0 --tag raw --raw --threads 1",
    "c2": f"{env} python run_retraining.py --threads 1; "
          f"{env} python run_elliptic.py --models {V} --seeds 0-2 --tag ablation2 --threads 1; "
          f"{env} python run_label_efficiency.py --threads 1",
}
out = subprocess.run("ps aux | grep -c '[r]un_'", shell=True, capture_output=True, text=True).stdout.strip()
if out != "0":
    print("jobs already running:", out)
else:
    for k, c in jobs.items():
        subprocess.Popen(f"cd /content/rtx/experiments && nohup sh -c '{c}' >> /content/{k}.log 2>&1 &",
                         shell=True, start_new_session=True)
    print("launched")
