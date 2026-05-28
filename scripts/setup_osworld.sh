#!/usr/bin/env bash
set -euo pipefail

ROOT=""
CLONE=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_osworld.sh --root /path/to/OSWorld
  bash scripts/setup_osworld.sh --root /path/to/OSWorld --clone

This script records OSWORLD_ROOT in .env and validates that the OSWorld
checkout has the files needed by the LearnWeak wrappers.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="${2:?missing value for --root}"
      shift 2
      ;;
    --clone)
      CLONE=1
      shift
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

if [[ -z "$ROOT" ]]; then
  ROOT="${OSWORLD_ROOT:-}"
fi

if [[ -z "$ROOT" ]]; then
  echo "Set --root or export OSWORLD_ROOT." >&2
  exit 1
fi

if [[ "$CLONE" == "1" && ! -d "$ROOT/.git" ]]; then
  git clone https://github.com/xlang-ai/osworld "$ROOT"
fi

if [[ ! -d "$ROOT" ]]; then
  echo "OSWorld root does not exist: $ROOT" >&2
  echo "Clone OSWorld first or rerun with --clone." >&2
  exit 1
fi

REQUIRED=(
  "lib_run_single.py"
  "desktop_env"
  "mm_agents"
)

for rel in "${REQUIRED[@]}"; do
  if [[ ! -e "$ROOT/$rel" ]]; then
    echo "Missing required OSWorld file: $ROOT/$rel" >&2
    exit 1
  fi
done

printf 'OSWORLD_ROOT=%s\n' "$(cd "$ROOT" && pwd)" > "$REPO_ROOT/.env"
echo "Wrote $REPO_ROOT/.env with OSWORLD_ROOT=$(cd "$ROOT" && pwd)"
echo "Run 'source $REPO_ROOT/.env' before using LearnWeak scripts."
