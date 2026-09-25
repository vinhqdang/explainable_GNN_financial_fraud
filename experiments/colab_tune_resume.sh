#!/bin/sh
# Usage: [VM=a|b] colab_tune_resume.sh SESSION -- (re)start the tuning study on a Colab VM.
# Uploads the local tuning records first, so finished configurations are skipped.
set -e
export PATH=$HOME/.local/bin:$PATH
S=$1; HERE=$(cd $(dirname "$0") && pwd)
VM=${VM:-a}
[ $VM = a ] && M="rtxgnn gas sage gcn tgn fraudre" || M="rtxgnn_b gat sefraud tgat"
timeout 1200 colab exec -s $S -f $HERE/colab_remote_setup.py --timeout 1180
FILES=""
for m in $M; do
  for f in tuning_$m.jsonl tuned_$m.jsonl; do
    FILES="$FILES $f"
    [ -s $HERE/../results/colab_live/$f ] && timeout 120 colab upload -s $S $HERE/../results/colab_live/$f /content/rtx/results/$f
  done
done
cp $HERE/colab_tune_launch.py /tmp/colab_tune_launch_$VM.py && sed -i "s/^vm = .*/vm = \"$VM\"/" /tmp/colab_tune_launch_$VM.py
timeout 120 colab exec -s $S -f /tmp/colab_tune_launch_$VM.py --timeout 100
DST=$HERE/../results/colab_live COLAB_SESSION=$S SYNC_EVERY=60 FILES="$FILES" LOGS="t1 t2" LOGPREFIX=${VM}_ nohup $HERE/colab_sync.sh $VM > /tmp/logs/colab_sync_$VM.log 2>&1 &
echo resumed on $S
