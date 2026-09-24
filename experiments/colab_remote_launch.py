"""Executed on the Colab VM by colab_resume.sh: (re)start the GPU job queues.
All scripts skip runs whose records already exist, so relaunching resumes."""
import subprocess
env = "RTX_DATA=/content/data RTX_DEVICE=cuda PYTHONPATH=.. MALLOC_MMAP_THRESHOLD_=65536"
V = ("rtxgnn:full,rtxgnn:time2vec_abs,rtxgnn:no_suf,rtxgnn:no_nec,rtxgnn:no_curriculum,rtxgnn:focal_loss,"
     "rtxgnn:sparsity_x4,rtxgnn:no_sparsity_drop0,rtxgnn:no_sparsity_drop5,rtxgnn:no_sparsity_wd1e3,"
     "rtxgnn:dim32,rtxgnn:dim128,rtxgnn:layers1")
jobs = {
    "p1": f"{env} python run_label_efficiency.py --threads 1",
    "p3": f"{env} python run_elliptic.py --models evolvegcn --seeds 0-9 --tag evo --threads 1; {env} python run_elliptic.py --models mlp,gcn,sage,gat,sefraud,rtxgnn --seeds 0 --tag raw --raw --threads 1",
    "p2": f"{env} python run_retraining.py --threads 1; {env} python run_elliptic.py --models {V} --seeds 0-2 --tag ablation2 --threads 1",
}
for k, c in jobs.items():
    subprocess.Popen(f"cd /content/rtx/experiments && nohup sh -c '{c}' >> /content/{k}.log 2>&1 &", shell=True, start_new_session=True)
print("launched")
