#!/bin/sh
# Downloads predictions and checkpoints written on the Colab VM (needed by the
# operational table, the explanation study and the case studies).
export PATH=$HOME/.local/bin:$PATH
S=$1; R=$(cd $(dirname "$0")/../results && pwd)
for sub in preds ckpt; do
  mkdir -p $R/$sub
  for f in $(timeout 120 colab ls -s $S /content/rtx/results/$sub 2>/dev/null | grep -E "\.(npy|pt)$"); do
    [ -s $R/$sub/$f ] || timeout 300 colab download -s $S /content/rtx/results/$sub/$f $R/$sub/$f >/dev/null 2>&1
  done
done
