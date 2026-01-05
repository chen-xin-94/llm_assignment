#!/usr/bin/env bash
set -euo pipefail

# Default session name is "train"
SESSION="${1:-train}"

# Window names (optional, but helpful)
NAMES=(
  "lora-original"
  "lora-dropped"
  "lora-pruned"
  "original"
  "dropped"
  "pruned"
)

# Each window: first run activation, then run training command
ACTIVATE_CMD="source .venv/bin/activate"

TRAIN_CMDS=(
  "CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/original_lora.yaml"
  "CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/dropped_lora.yaml"
  "CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/pruned_lora.yaml"
  "CUDA_VISIBLE_DEVICES=4 python scripts/train.py --config configs/original.yaml"
  "CUDA_VISIBLE_DEVICES=5 python scripts/train.py --config configs/dropped.yaml"
  "CUDA_VISIBLE_DEVICES=7 python scripts/train.py --config configs/pruned.yaml"
)

# If you want the window to stay open after training finishes:
KEEP_OPEN=1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n "${NAMES[0]}"

run_in_window() {
  local target="$1"
  local train_cmd="$2"

  # Step 1: activate env (press Enter)
  tmux send-keys -t "$target" "$ACTIVATE_CMD" C-m

  # Step 2: run training (press Enter)
  if [[ "$KEEP_OPEN" -eq 1 ]]; then
    tmux send-keys -t "$target" "$train_cmd; echo; echo '--- done ---'; exec bash" C-m
  else
    tmux send-keys -t "$target" "$train_cmd" C-m
  fi
}

run_in_window "${SESSION}:0" "${TRAIN_CMDS[0]}"

for i in {1..5}; do
  tmux new-window -t "$SESSION" -n "${NAMES[$i]}"
  run_in_window "${SESSION}:$i" "${TRAIN_CMDS[$i]}"
done

tmux select-window -t "${SESSION}:0"
tmux attach -t "$SESSION"
