#!/bin/sh
# Re-creates the Colab session and resumes the GPU jobs whenever the VM is lost.
export PATH=$HOME/.local/bin:$PATH
HERE=$(cd $(dirname "$0") && pwd)
S=$1; N=${2:-3}
while true; do
  # the server-side session list is authoritative (status may show cached metadata)
  if ! timeout 90 colab sessions 2>&1 | grep -q "\[$S\]"; then
    echo "$(date +%T) session $S lost"
    for p in $(ps -eo pid,args | grep "[c]olab_sync.sh" | awk '{print $1}'); do kill $p; done
    S=gpu$N; N=$((N+1))
    if timeout 600 colab new -s $S --gpu T4 < /dev/null; then
      $HERE/colab_resume.sh $S && echo "$(date +%T) resumed on $S"
    fi
  fi
  [ -f /tmp/colab_done ] && break
  sleep 120
done
