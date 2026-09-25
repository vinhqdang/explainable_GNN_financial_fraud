#!/bin/sh
# Re-creates the Colab session and resumes the GPU jobs whenever the VM is lost
# or the sync heartbeat is older than 10 minutes.
export PATH=$HOME/.local/bin:$PATH
HERE=$(cd $(dirname "$0") && pwd)
S=$1; N=${2:-3}
while true; do
  stale=0
  if [ -f /tmp/logs/colab_sync${VM:+_${VM}_}.last ]; then
    age=$(( $(date +%s) - $(stat -c %Y /tmp/logs/colab_sync${VM:+_${VM}_}.last) ))
    [ $age -gt 600 ] && stale=1
  fi
  if [ $stale = 1 ] || ! timeout 90 colab sessions 2>&1 | grep -q "\\[$S\\]"; then
    echo "$(date +%T) session $S lost or stale"
    for p in $(pgrep -f "^/bin/sh .*colab_sync.sh ${VM:-}"); do kill $p; done
    timeout 120 colab stop -s $S < /dev/null >/dev/null 2>&1
    S=${PREFIX:-gpu}$N; N=$((N+1))
    touch /tmp/logs/colab_sync${VM:+_${VM}_}.last
    if timeout 600 colab new -s $S --gpu T4 < /dev/null; then
      for i in 1 2 3; do
        $HERE/${RESUME:-colab_resume.sh} $S && { echo "$(date +%T) resumed on $S"; break; }
        sleep 30
      done
    fi
  fi
  [ -f /tmp/colab_done ] && break
  sleep 120
done
