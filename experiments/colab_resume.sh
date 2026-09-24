#!/bin/sh
# Usage: colab_resume.sh SESSION   -- (re)start the Colab GPU jobs after a VM loss.
# 1) set up the VM, 2) upload the local copies of the Colab-produced records
# (so finished runs are skipped), 3) relaunch the queues, 4) start the sync loop.
set -e
export PATH=$HOME/.local/bin:$PATH
S=$1; HERE=$(cd $(dirname "$0") && pwd)
timeout 1200 colab exec -s $S -f $HERE/colab_remote_setup.py --timeout 1180
for f in label_efficiency.jsonl retraining.jsonl elliptic_ablation2.jsonl elliptic_evo.jsonl elliptic_raw.jsonl; do
  [ -s $HERE/../results/$f ] && timeout 120 colab upload -s $S $HERE/../results/$f /content/rtx/results/$f
done
timeout 120 colab exec -s $S -f $HERE/colab_remote_launch.py --timeout 100
COLAB_SESSION=$S SYNC_EVERY=60 nohup $HERE/colab_sync.sh > /tmp/logs/colab_sync.log 2>&1 &
echo resumed on $S
