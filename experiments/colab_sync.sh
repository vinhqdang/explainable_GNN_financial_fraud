#!/bin/sh
# Pulls the result records produced on the Colab VM every few minutes, so that
# nothing but the run in progress is lost if the VM dies.
export PATH=$HOME/.local/bin:$PATH
S=${COLAB_SESSION:-gpu1}
DST=${DST:-$(dirname "$0")/../results}
while true; do
  for f in ${FILES:-label_efficiency.jsonl retraining.jsonl elliptic_ablation2.jsonl elliptic_evo.jsonl elliptic_raw.jsonl elliptic_main_gpu.jsonl tfinance.jsonl}; do
    timeout 120 colab download -s $S /content/rtx/results/$f $DST/$f.colab.tmp >/dev/null 2>&1 && \
      [ -s $DST/$f.colab.tmp ] && mv $DST/$f.colab.tmp $DST/$f
    rm -f $DST/$f.colab.tmp
  done
  $(dirname "$0")/colab_fetch_artifacts.sh $S
  for p in ${LOGS:-c1 c2 c3}; do timeout 60 colab download -s $S /content/$p.log /tmp/logs/colab_${LOGPREFIX}$p.log >/dev/null 2>&1; done
  timeout 60 colab sessions 2>&1 | grep -q "\[$S\]" && date +%T > /tmp/logs/colab_sync${LOGPREFIX:+_$LOGPREFIX}.last
  sleep ${SYNC_EVERY:-180}
done
