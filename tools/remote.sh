#!/usr/bin/env bash
# Drive training on a GPU laptop over SSH. Run this on the Mac; setup steps are in docs/TRAINING.md.
#
#   tools/remote.sh sync   <host>                          copy the code to ~/nova-train
#   tools/remote.sh setup  <host>                          sync, install packages, check GPU and speed
#   tools/remote.sh start  <host> <config> [train args]    start or resume training in tmux
#   tools/remote.sh status <host> [lines]                  GPU, memory and the latest training log
#   tools/remote.sh stop   <host>                          save a checkpoint and stop
#   tools/remote.sh board  <host> [port]                   TensorBoard at http://localhost:<port>
#   tools/remote.sh fetch  <host>                          copy runs/ back to runs/<host>/
#
# <host> is an alias from ~/.ssh/config. One training run per laptop.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR=nova-train  # inside the remote home directory
SESSION=nova-train     # tmux session name

usage() { sed -n '2,/^set -/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }

# The login shell on a laptop may be fish, which joins and re-parses arguments. Allow only characters
# that mean the same thing to every shell.
check_word() {
  case $1 in
    "" | *[!A-Za-z0-9_./=,:@+-]*) die "unsupported characters in argument '$1'" ;;
  esac
}

# Run the bash script given on stdin on the host: rbash <host> [args for the script]
rbash() {
  local host=$1 arg
  shift
  for arg in "$@"; do check_word "$arg"; done
  ssh "$host" bash -s -- "$@"
}

require_idle() {
  rbash "$1" "$SESSION" <<'EOF'
if tmux has-session -t "=$1" 2>/dev/null; then
  echo "error: training is running on this laptop; stop it first" >&2
  exit 1
fi
EOF
}

do_sync() {
  local host=$1
  require_idle "$host"
  # Excluded paths are also protected from --delete, so the laptop keeps its .venv and runs/.
  rsync -az --delete \
    --exclude=.venv --exclude=/runs --exclude=__pycache__ --exclude=.pytest_cache \
    --exclude=.DS_Store --exclude=/_vizdoom.ini --exclude=.env \
    "$ROOT/" "$host:$REMOTE_DIR/"
  echo "synced $ROOT to $host:~/$REMOTE_DIR"
}

do_setup() {
  local host=$1
  do_sync "$host"
  rbash "$host" "$REMOTE_DIR" <<'EOF'
set -eu
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/$1"
for tool in uv tmux; do
  command -v "$tool" >/dev/null || { echo "error: $tool is not installed (see docs/TRAINING.md)" >&2; exit 1; }
done
echo "installing Python packages (the first time downloads several GB)"
UV_HTTP_TIMEOUT=600 uv sync </dev/null
.venv/bin/python tools/check_setup.py </dev/null
EOF
}

do_start() {
  local host=$1 config=${2:-}
  [ -n "$config" ] || die "usage: start <host> <config> [train args]"
  shift 2
  [ -f "$ROOT/$config" ] || die "no such config: $config (give a path relative to the repository)"

  # The run name decides the log file and the runs/ folder: --run-name wins over the config's name.
  local name="" previous="" arg
  for arg in "$@"; do
    case $arg in --run-name=*) name=${arg#--run-name=} ;; esac
    [ "$previous" = --run-name ] && name=$arg
    previous=$arg
  done
  [ -n "$name" ] || name=$(sed -n 's/^name:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$ROOT/$config")
  [ -n "$name" ] || name=run

  rbash "$host" "$REMOTE_DIR" "$SESSION" "$name" "$config" "$@" <<'EOF'
set -eu
dir="$HOME/$1" session=$2 name=$3 config=$4
shift 4
cd "$dir"
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "error: training is already running on this laptop" >&2
  exit 1
fi
[ -x .venv/bin/python ] || { echo "error: run setup first" >&2; exit 1; }
[ -f "$config" ] || { echo "error: $config is not on the laptop; run sync" >&2; exit 1; }
mkdir -p runs
log="runs/$name.console.log"
{
  printf 'cd %q || exit 1\n' "$dir"
  echo 'export PYTHONUNBUFFERED=1'
  printf '%q ' .venv/bin/python -m training.train --config "$config" "$@"
  # tee -i ignores Ctrl+C, so the pipe stays open while training saves its checkpoint.
  printf '2>&1 | tee -i -a %q\n' "$log"
} > runs/launch.sh
echo "=== start $(date) : $config $*" >> "$log"
tmux new-session -d -s "$session" "bash '$dir/runs/launch.sh'"
sleep 15
tail -n 8 "$log"
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "running; log: $dir/$log"
else
  echo "error: training exited during start-up (log above)" >&2
  exit 1
fi
EOF
}

do_status() {
  local host=$1 lines=${2:-12}
  rbash "$host" "$REMOTE_DIR" "$SESSION" "$lines" <<'EOF'
cd "$HOME/$1" 2>/dev/null || { echo "nothing synced yet"; exit 0; }
if tmux has-session -t "=$2" 2>/dev/null; then echo "training: running"; else echo "training: not running"; fi
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv </dev/null
fi
if command -v free >/dev/null; then free -h </dev/null; fi
latest=$(ls -t runs/*.console.log 2>/dev/null | head -n 1)
if [ -n "$latest" ]; then echo "--- $latest"; tail -n "$3" "$latest"; fi
EOF
}

do_stop() {
  local host=$1
  rbash "$host" "$REMOTE_DIR" "$SESSION" <<'EOF'
cd "$HOME/$1" || exit 1
session=$2
if ! tmux has-session -t "=$session" 2>/dev/null; then echo "training is not running"; exit 0; fi
tmux send-keys -t "=$session:" C-c
echo "asked training to save and stop"
waited=0
while tmux has-session -t "=$session" 2>/dev/null; do
  if [ "$waited" -ge 180 ]; then
    echo "error: still running after ${waited}s; check status" >&2
    exit 1
  fi
  sleep 2
  waited=$((waited + 2))
done
latest=$(ls -t runs/*.console.log 2>/dev/null | head -n 1)
tail -n 5 "$latest"
ls -l runs/*/ckpt_latest.pt
EOF
}

do_board() {
  local host=$1 port=${2:-16006}
  check_word "$port"
  echo "open http://localhost:$port  (Ctrl+C here closes it; training is not affected)"
  ssh -t -L "$port:localhost:6006" "$host" "bash -c 'cd ~/$REMOTE_DIR && .venv/bin/tensorboard --logdir runs --port 6006'"
}

do_fetch() {
  local host=$1
  mkdir -p "$ROOT/runs/$host"
  rsync -az --exclude='*.tmp' --exclude=launch.sh "$host:$REMOTE_DIR/runs/" "$ROOT/runs/$host/"
  echo "copied to runs/$host/"
}

command=${1:-}
case $command in
  sync | setup | start | status | stop | board | fetch) ;;
  "" | -h | --help | help) usage; exit 0 ;;
  *) usage >&2; die "unknown command '$command'" ;;
esac
shift
[ $# -ge 1 ] || die "usage: $command <host>"
check_word "$1"
case $1 in */* | -*) die "'$1' is not a host alias" ;; esac
"do_$command" "$@"
