#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${DOMAIN:-gimp}"
STEP="${STEP:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

OSWORLD_ROOT="${OSWORLD_ROOT:?Set OSWORLD_ROOT or run scripts/setup_osworld.sh first.}"
DATA_ROOT="${LEARNWEAK_DATA_ROOT:-$REPO_ROOT/learnweak_gen/data}"
GENERATED_DATA_ROOT="${LEARNWEAK_GENERATED_DATA_ROOT:-$REPO_ROOT/learnweak_gen/data/synthetic_evocua}"
ROLLOUT_ROOT="$REPO_ROOT/learnweak_gen/rollouts"

TEACHER_MODEL="${TEACHER_MODEL:-meituan/EvoCUA-32B-20260105}"
STUDENT_MODEL="${STUDENT_MODEL:-vllm_evocua-8b}"
STUDENT_VLLM_URL="${STUDENT_VLLM_URL:-${VLLM_BASE_URL:-http://localhost:7703}}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ "$STEP" == "1" ]]; then
  task_file="$DATA_ROOT/seed/test_${DOMAIN}.json"
  task_config_dir="$DATA_ROOT/seed/examples"
else
  task_file="$GENERATED_DATA_ROOT/iter${STEP}/test_${DOMAIN}.json"
  task_config_dir="$GENERATED_DATA_ROOT/iter${STEP}/examples"
fi

teacher_result_dir="$ROLLOUT_ROOT/dataset_generation/$DOMAIN/evocua-32b_synthetic_step${STEP}/pyautogui/screenshot/$TEACHER_MODEL"
save_dir="$ROLLOUT_ROOT/dataset_generation/$DOMAIN/evocua-32b_synthetic_step${STEP}_student"

python "$REPO_ROOT/learnweak_gen/run_student_on_teacher_trajectory.py" \
  --osworld-root "$OSWORLD_ROOT" \
  --domain "$DOMAIN" \
  --task-file "$task_file" \
  --task-config-dir "$task_config_dir" \
  --teacher-result-dir "$teacher_result_dir" \
  --save-dir "$save_dir" \
  --student-model "$STUDENT_MODEL" \
  --vllm-base-url "$STUDENT_VLLM_URL" \
  --num-workers "$NUM_WORKERS"
