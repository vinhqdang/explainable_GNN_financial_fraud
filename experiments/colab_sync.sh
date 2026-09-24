#!/bin/sh
# Pulls the result records produced on the Colab VM every few minutes, so that
# nothing but the run in progress is lost if the VM dies.
export PATH=$HOME/.local/bin:$PATH
S=${COLAB_SESSION:-gpu1}
DST=$(dirname "$0")/../results
while true; do
  for f in label_efficiency.jsonl retraining.jsonl elliptic_ablation2.jsonl; do
    timeout 120 colab download -s $S /content/rtx/results/$f $DST/$f.colab.tmp >/dev/null 2>&1 && \
      [ -s $DST/$f.colab.tmp ] && mv $DST/$f.colab.tmp $DST/$f
    rm -f $DST/$f.colab.tmp
  done
  for p in p1 p2; do timeout 60 colab download -s $S /content/$p.log /tmp/logs/colab_$p.log >/dev/null 2>&1; done
  date +%T > /tmp/logs/colab_sync.last
  sleep ${SYNC_EVERY:-180}
done
