#!/usr/bin/env bash
set -euo pipefail

DOMAIN="${DOMAIN:-gimp}"
ITERATIONS="${ITERATIONS:-5}"
STUDENT_MODEL="${STUDENT_MODEL:-vllm_evocua-8b}"
TEACHER_MODEL="${TEACHER_MODEL:-meituan/EvoCUA-32B-20260105}"
STUDENT_VLLM_URL="${STUDENT_VLLM_URL:-http://localhost:7703}"
NUM_WORKERS="${NUM_WORKERS:-4}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_teacher_replay_gimp.sh [options]

Options:
  --domain DOMAIN                 Target domain. Default: gimp
  --iterations N                  Last iteration to replay. Default: 5
  --num-workers N                 Parallel replay workers. Default: 4
  -h, --help                      Show this help

Environment:
  STUDENT_MODEL                   Student served model name. Default: vllm_evocua-8b
  TEACHER_MODEL                   Teacher model name used in result paths.
                                  Default: meituan/EvoCUA-32B-20260105
  STUDENT_VLLM_URL                Student vLLM base URL. Default: http://localhost:7703
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain)
      DOMAIN="${2:?missing value for --domain}"
      shift 2
      ;;
    --iterations)
      ITERATIONS="${2:?missing value for --iterations}"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:?missing value for --num-workers}"
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLAY="$REPO_ROOT/scripts/run_pair_inference_gimp.sh"

for step in $(seq 1 "$ITERATIONS"); do
  echo "[LearnWeak-DPO] teacher replay: domain=$DOMAIN step=$step"
  DOMAIN="$DOMAIN" \
  STEP="$step" \
  STUDENT_MODEL="$STUDENT_MODEL" \
  TEACHER_MODEL="$TEACHER_MODEL" \
  STUDENT_VLLM_URL="$STUDENT_VLLM_URL" \
  NUM_WORKERS="$NUM_WORKERS" \
  "$REPLAY"
done

echo "[LearnWeak-DPO] teacher replay done: domain=$DOMAIN iterations=$ITERATIONS"
