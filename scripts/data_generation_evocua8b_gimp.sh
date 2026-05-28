#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

DOMAIN="${DOMAIN:-gimp}"
STEP="${STEP:-2}"
STUDENT_MODEL="${STUDENT_MODEL:-vllm_evocua-8b}"
TEACHER_MODEL="${TEACHER_MODEL:-meituan/EvoCUA-32B-20260105}"
STUDENT_VLLM_URL="${STUDENT_VLLM_URL:-http://localhost:7703}"
TEACHER_VLLM_URL="${TEACHER_VLLM_URL:-http://localhost:7793}"
SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"
ITERATIONS="${ITERATIONS:-5}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_HISTORY_TURNS="${MAX_HISTORY_TURNS:-4}"
TEMPERATURE="${TEMPERATURE:-0.01}"
UNIT=""

DATA_ROOT="${LEARNWEAK_DATA_ROOT:-$REPO_ROOT/learnweak_gen/data}"
GENERATED_DATA_ROOT="${LEARNWEAK_GENERATED_DATA_ROOT:-$REPO_ROOT/learnweak_gen/data/synthetic_evocua}"
ROLLOUT_ROOT="$REPO_ROOT/learnweak_gen/rollouts"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data_generation_evocua8b_gimp.sh [options]
  bash scripts/data_generation_evocua8b_gimp.sh --unit UNIT [--step N]

Full pipeline options:
  --domain DOMAIN                 Target domain. Default: gimp
  --iterations N                  Last generation iteration. Default: 5

Model/server environment:
  STUDENT_MODEL                   Student served model name. Default: vllm_evocua-8b
  TEACHER_MODEL                   Teacher served model name. Default: meituan/EvoCUA-32B-20260105
  STUDENT_VLLM_URL                Student vLLM base URL. Default: http://localhost:7703
  TEACHER_VLLM_URL                Teacher vLLM base URL. Default: http://localhost:7793
  --skip-vllm-check               Skip /v1/models connectivity checks

Single-unit/resume options:
  --unit UNIT                     Run one unit and exit
  --step N                        Iteration number for iter_* units. Default: 2

Units:
  seed_student_inference, seed_teacher_inference, seed_student_verify,
  seed_teacher_verify, seed_set_strategy, seed_find_unique_screenshots,
  seed_rank_screenshots, seed_gen_with_fail_report,
  seed_gen_without_fail_report, seed_prepare_next,
  iter_student_inference, iter_teacher_inference, iter_student_verify,
  iter_teacher_verify, iter_set_strategy, iter_find_unique_screenshots,
  iter_rank_screenshots, iter_gen_with_fail_report,
  iter_gen_without_fail_report, iter_prepare_next

Notes:
  OSWORLD_ROOT is required only for *_inference units.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)
      DOMAIN="${2:?missing value for --domain}"
      shift 2
      ;;
    --skip-vllm-check)
      SKIP_VLLM_CHECK=1
      shift
      ;;
    --iterations)
      ITERATIONS="${2:?missing value for --iterations}"
      shift 2
      ;;
    --step)
      STEP="${2:?missing value for --step}"
      shift 2
      ;;
    --unit)
      UNIT="${2:?missing value for --unit}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
  echo "--iterations must be a positive integer." >&2
  exit 1
fi

if ! [[ "$STEP" =~ ^[0-9]+$ ]] || [[ "$STEP" -lt 1 ]]; then
  echo "--step must be a positive integer." >&2
  exit 1
fi

seed_eval_dir="$DATA_ROOT/seed"
seed_instruction_dir="$seed_eval_dir/examples/$DOMAIN"
manual_configs_dir="$DATA_ROOT/manual/$DOMAIN"

iter_eval_dir="$GENERATED_DATA_ROOT/iter$STEP"
iter_instruction_dir="$iter_eval_dir/examples/$DOMAIN"

result_base="$ROLLOUT_ROOT/dataset_generation/$DOMAIN"
seed_student_result="$result_base/evocua-8b_synthetic_step1"
seed_teacher_result="$result_base/evocua-32b_synthetic_step1"
iter_student_result="$result_base/evocua-8b_synthetic_step$STEP"
iter_teacher_result="$result_base/evocua-32b_synthetic_step$STEP"

student_leaf="pyautogui/screenshot/$STUDENT_MODEL"
teacher_leaf="pyautogui/screenshot/$TEACHER_MODEL"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
}

unit_requires_osworld() {
  case "$1" in
    seed_student_inference|seed_teacher_inference|iter_student_inference|iter_teacher_inference)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

check_osworld() {
  OSWORLD_ROOT="${OSWORLD_ROOT:?Set OSWORLD_ROOT or run scripts/setup_osworld.sh first.}"
  require_dir "$OSWORLD_ROOT"
  require_file "$OSWORLD_ROOT/lib_run_single.py"
  require_dir "$OSWORLD_ROOT/desktop_env"
  require_dir "$OSWORLD_ROOT/mm_agents"
}

check_vllm_server() {
  local label="$1"
  local url="$2"
  local model="$3"

  if [[ "$SKIP_VLLM_CHECK" == "1" ]]; then
    return
  fi

  python - "$label" "$url" "$model" <<'PY'
import json
import sys
import urllib.request

label, raw_url, model = sys.argv[1:4]
base = raw_url.rstrip("/")
models_url = base + "/models" if base.endswith("/v1") else base + "/v1/models"

try:
    with urllib.request.urlopen(models_url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(
        f"{label} vLLM server check failed: {models_url}\n"
        f"  {exc}\n"
        "  Start the vLLM server or pass the correct --*-vllm-url."
    )

ids = []
for item in payload.get("data", []):
    if isinstance(item, dict) and item.get("id"):
        ids.append(str(item["id"]))

if model not in ids:
    raise SystemExit(
        f"{label} model was not found at {models_url}: {model}\n"
        f"  Available models: {', '.join(ids) if ids else '(none)'}\n"
        "  Set the served model name with STUDENT_MODEL or TEACHER_MODEL."
    )
PY
}

preflight_unit() {
  local unit="$1"
  require_dir "$DATA_ROOT/seed"
  require_dir "$DATA_ROOT/seed/examples/$DOMAIN"
  require_file "$DATA_ROOT/seed/test_${DOMAIN}.json"
  require_dir "$manual_configs_dir"

  if unit_requires_osworld "$unit"; then
    check_osworld
    case "$unit" in
      seed_student_inference|iter_student_inference)
        check_vllm_server "student" "$STUDENT_VLLM_URL" "$STUDENT_MODEL"
        ;;
      seed_teacher_inference|iter_teacher_inference)
        check_vllm_server "teacher" "$TEACHER_VLLM_URL" "$TEACHER_MODEL"
        ;;
    esac
  fi

  case "$unit" in
    iter_*)
      require_dir "$iter_eval_dir"
      require_dir "$iter_instruction_dir"
      require_file "$iter_eval_dir/test_${DOMAIN}.json"
      ;;
  esac
}

run_osworld_inference() {
  local model="$1"
  local vllm_url="$2"
  local result_dir="$3"
  local test_json="$4"
  local config_base="$5"

  mkdir -p "$REPO_ROOT/logs"
  (
    cd "$REPO_ROOT"
    VLLM_BASE_URL="$vllm_url" \
    python "$REPO_ROOT/learnweak_gen/run_osworld_evocua.py" \
      --headless \
      --provider_name docker \
      --api_backend vllm \
      --observation_type screenshot \
      --model "$model" \
      --result_dir "$result_dir" \
      --test_all_meta_path "$test_json" \
      --max_steps "$MAX_STEPS" \
      --num_envs "$NUM_ENVS" \
      --temperature "$TEMPERATURE" \
      --max_history_turns "$MAX_HISTORY_TURNS" \
      --coordinate_type relative \
      --resize_factor 32 \
      --prompt_style S2 \
      --log_level INFO \
      --test_config_base_dir "$config_base"
  )
}

run_unit() {
  local unit="$1"
  preflight_unit "$unit"

  case "$unit" in
    seed_student_inference)
      run_osworld_inference "$STUDENT_MODEL" "$STUDENT_VLLM_URL" "$seed_student_result" "$seed_eval_dir/test_${DOMAIN}.json" "$seed_eval_dir"
      ;;
    seed_teacher_inference)
      run_osworld_inference "$TEACHER_MODEL" "$TEACHER_VLLM_URL" "$seed_teacher_result" "$seed_eval_dir/test_${DOMAIN}.json" "$seed_eval_dir"
      ;;
    seed_student_verify)
      python "$REPO_ROOT/learnweak_gen/teacher_verify_results.py" --traj-dir "$seed_student_result/$student_leaf/$DOMAIN" --instruction-dir "$seed_instruction_dir" --out "$seed_student_result/$student_leaf/verify_results.json"
      ;;
    seed_teacher_verify)
      python "$REPO_ROOT/learnweak_gen/teacher_verify_results.py" --traj-dir "$seed_teacher_result/$teacher_leaf/$DOMAIN" --instruction-dir "$seed_instruction_dir" --out "$seed_teacher_result/$teacher_leaf/verify_results.json"
      ;;
    seed_set_strategy)
      python "$REPO_ROOT/learnweak_gen/teacher_set_strategy.py" --teacher "$seed_teacher_result/$teacher_leaf/verify_results.json" --student "$seed_student_result/$student_leaf/verify_results.json"
      ;;
    seed_find_unique_screenshots)
      python "$REPO_ROOT/learnweak_gen/find_unique_screenshot_samples.py" --folders "$seed_teacher_result/$teacher_leaf/$DOMAIN" "$seed_student_result/$student_leaf/$DOMAIN" --output-dir "$seed_student_result/$student_leaf"
      ;;
    seed_rank_screenshots)
      python "$REPO_ROOT/learnweak_gen/rank_screenshot_samples.py" --input-json "$seed_student_result/$student_leaf/sample_screenshots.json" --output-json "$seed_student_result/$student_leaf/final_screenshots.json"
      ;;
    seed_gen_with_fail_report)
      python "$REPO_ROOT/learnweak_gen/gen_new_queries.py" --domain "$DOMAIN" --configs-dir "$manual_configs_dir" --prior-instructions-dir "$seed_instruction_dir" --final-screenshots "$seed_student_result/$student_leaf/final_screenshots.json" --fail-report "$seed_student_result/$student_leaf/teacher_pass_student_fail_report.json"
      ;;
    seed_gen_without_fail_report)
      python "$REPO_ROOT/learnweak_gen/gen_new_queries.py" --domain "$DOMAIN" --configs-dir "$manual_configs_dir" --prior-instructions-dir "$seed_instruction_dir" --final-screenshots "$seed_student_result/$student_leaf/final_screenshots.json" --no-fail-report
      ;;
    seed_prepare_next)
      python "$REPO_ROOT/learnweak_gen/prepare_run_for_new_queries.py" "$seed_student_result/$student_leaf/new_queries_per_config_nofail.json" "$seed_student_result/$student_leaf/new_queries_per_config.json" --out-dir "$GENERATED_DATA_ROOT/iter2/examples/$DOMAIN" --test-json "$GENERATED_DATA_ROOT/iter2/test_${DOMAIN}.json" --domain "$DOMAIN"
      ;;
    iter_student_inference)
      run_osworld_inference "$STUDENT_MODEL" "$STUDENT_VLLM_URL" "$iter_student_result" "$iter_eval_dir/test_${DOMAIN}.json" "$iter_eval_dir"
      ;;
    iter_teacher_inference)
      run_osworld_inference "$TEACHER_MODEL" "$TEACHER_VLLM_URL" "$iter_teacher_result" "$iter_eval_dir/test_${DOMAIN}.json" "$iter_eval_dir"
      ;;
    iter_student_verify)
      python "$REPO_ROOT/learnweak_gen/teacher_verify_results.py" --traj-dir "$iter_student_result/$student_leaf/$DOMAIN" --instruction-dir "$iter_instruction_dir" --out "$iter_student_result/$student_leaf/verify_results.json"
      ;;
    iter_teacher_verify)
      python "$REPO_ROOT/learnweak_gen/teacher_verify_results.py" --traj-dir "$iter_teacher_result/$teacher_leaf/$DOMAIN" --instruction-dir "$iter_instruction_dir" --out "$iter_teacher_result/$teacher_leaf/verify_results.json"
      ;;
    iter_set_strategy)
      python "$REPO_ROOT/learnweak_gen/teacher_set_strategy.py" --teacher "$iter_teacher_result/$teacher_leaf/verify_results.json" --student "$iter_student_result/$student_leaf/verify_results.json"
      ;;
    iter_find_unique_screenshots)
      python "$REPO_ROOT/learnweak_gen/find_unique_screenshot_samples.py" --folders "$iter_teacher_result/$teacher_leaf/$DOMAIN" "$iter_student_result/$student_leaf/$DOMAIN" --output-dir "$iter_student_result/$student_leaf"
      ;;
    iter_rank_screenshots)
      python "$REPO_ROOT/learnweak_gen/rank_screenshot_samples.py" --input-json "$iter_student_result/$student_leaf/sample_screenshots.json" --output-json "$iter_student_result/$student_leaf/final_screenshots.json"
      ;;
    iter_gen_with_fail_report)
      python "$REPO_ROOT/learnweak_gen/gen_new_queries.py" --domain "$DOMAIN" --configs-dir "$manual_configs_dir" --prior-instructions-dir "$iter_instruction_dir" --final-screenshots "$iter_student_result/$student_leaf/final_screenshots.json" --fail-report "$iter_student_result/$student_leaf/teacher_pass_student_fail_report.json"
      ;;
    iter_gen_without_fail_report)
      python "$REPO_ROOT/learnweak_gen/gen_new_queries.py" --domain "$DOMAIN" --configs-dir "$manual_configs_dir" --prior-instructions-dir "$iter_instruction_dir" --final-screenshots "$iter_student_result/$student_leaf/final_screenshots.json" --no-fail-report
      ;;
    iter_prepare_next)
      next_step=$((STEP + 1))
      python "$REPO_ROOT/learnweak_gen/prepare_run_for_new_queries.py" "$iter_student_result/$student_leaf/new_queries_per_config_nofail.json" "$iter_student_result/$student_leaf/new_queries_per_config.json" --out-dir "$GENERATED_DATA_ROOT/iter${next_step}/examples/$DOMAIN" --test-json "$GENERATED_DATA_ROOT/iter${next_step}/test_${DOMAIN}.json" --domain "$DOMAIN"
      ;;
    *)
      echo "Unknown unit: $unit" >&2
      usage >&2
      exit 1
      ;;
  esac
}

run_step() {
  local unit="$1"
  echo "[LearnWeak-GEN] $unit"
  run_unit "$unit"
}

run_seed() {
  echo "[LearnWeak-GEN] seed: student/teacher inference"
  run_step seed_student_inference
  run_step seed_teacher_inference

  echo "[LearnWeak-GEN] seed: verification and weakness analysis"
  run_step seed_student_verify
  run_step seed_teacher_verify
  run_step seed_set_strategy

  echo "[LearnWeak-GEN] seed: next-task generation"
  run_step seed_find_unique_screenshots
  run_step seed_rank_screenshots
  run_step seed_gen_with_fail_report
  run_step seed_gen_without_fail_report
  run_step seed_prepare_next
}

run_iter() {
  STEP="$1"
  iter_eval_dir="$GENERATED_DATA_ROOT/iter$STEP"
  iter_instruction_dir="$iter_eval_dir/examples/$DOMAIN"
  iter_student_result="$result_base/evocua-8b_synthetic_step$STEP"
  iter_teacher_result="$result_base/evocua-32b_synthetic_step$STEP"

  echo "[LearnWeak-GEN] iter${STEP}: student/teacher inference"
  run_step iter_student_inference
  run_step iter_teacher_inference

  echo "[LearnWeak-GEN] iter${STEP}: verification and weakness analysis"
  run_step iter_student_verify
  run_step iter_teacher_verify
  run_step iter_set_strategy

  if [[ "$STEP" -lt "$ITERATIONS" ]]; then
    echo "[LearnWeak-GEN] iter${STEP}: next-task generation"
    run_step iter_find_unique_screenshots
    run_step iter_rank_screenshots
    run_step iter_gen_with_fail_report
    run_step iter_gen_without_fail_report
    run_step iter_prepare_next
  fi
}

echo "[LearnWeak-GEN] domain=$DOMAIN student=$STUDENT_MODEL teacher=$TEACHER_MODEL"
echo "[LearnWeak-GEN] student_vllm_url=$STUDENT_VLLM_URL teacher_vllm_url=$TEACHER_VLLM_URL"
echo "[LearnWeak-GEN] generated_data_root=$GENERATED_DATA_ROOT rollout_root=$ROLLOUT_ROOT"

if [[ -n "$UNIT" ]]; then
  echo "[LearnWeak-GEN] unit=$UNIT step=$STEP"
  run_unit "$UNIT"
  exit 0
fi

run_seed
for step in $(seq 2 "$ITERATIONS"); do
  run_iter "$step"
done

echo "[LearnWeak-GEN] done: domain=$DOMAIN iterations=$ITERATIONS"
