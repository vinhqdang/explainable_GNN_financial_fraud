#!/bin/sh
# Usage: colab_tune_resume.sh SESSION -- (re)start the tuning study on a Colab VM.
# Uploads the local tuning records first, so finished configurations are skipped.
set -e
export PATH=$HOME/.local/bin:$PATH
S=$1; HERE=$(cd $(dirname "$0") && pwd)
M="rtxgnn gas sage gcn tgn fraudre tgat gat sefraud"
timeout 1200 colab exec -s $S -f $HERE/colab_remote_setup.py --timeout 1180
FILES=""
for m in $M; do
  for f in tuning_$m.jsonl tuned_$m.jsonl; do
    FILES="$FILES $f"
    [ -s $HERE/../results/colab_live/$f ] && timeout 120 colab upload -s $S $HERE/../results/colab_live/$f /content/rtx/results/$f
  done
done
timeout 120 colab exec -s $S -f $HERE/colab_tune_launch.py --timeout 100
DST=$HERE/../results/colab_live COLAB_SESSION=$S SYNC_EVERY=60 FILES="$FILES" LOGS="t1 t2" nohup $HERE/colab_sync.sh > /tmp/logs/colab_sync.log 2>&1 &
echo resumed on $S
