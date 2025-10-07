#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (script is in scripts/)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
LOG_DIR_DEFAULT=${REPO_ROOT}/logs/go1_interfaces

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options] [AGENT[:CONFIG]] ...

Examples:
  - Default two agents (go1_0, go1_1):
      $(basename "$0")
  - Single agent with inferred config path:
      $(basename "$0") go1_0
    (uses: \${REPO_ROOT}/sim2real/config/go1_0.yaml)
  - Two agents with explicit configs:
      $(basename "$0") go1_0:${REPO_ROOT}/sim2real/config/go1_0.yaml go1_1:${REPO_ROOT}/sim2real/config/go1_1.yaml

Options:
  --no-tail            Do not tail logs after launching.
  --dry-run            Print what would be run, but do not start processes.
  --python-bin=PATH    Python interpreter to use (default: ${PYTHON_BIN}).
  --log-dir=DIR        Log directory (default: ${LOG_DIR_DEFAULT}).
  -h, --help           Show this help.

Notes:
  - If CONFIG is omitted for an AGENT, it defaults to:
      \${REPO_ROOT}/sim2real/config/AGENT.yaml
  - You can run this script from any directory.
EOF
}

NO_TAIL=0
DRY_RUN=0
LOG_DIR="${LOG_DIR_DEFAULT}"

# Parse options
ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --no-tail) NO_TAIL=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --python-bin=*) PYTHON_BIN="${arg#*=}" ;;
    --log-dir=*) LOG_DIR="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    --*) echo "Unknown option: ${arg}" >&2; usage; exit 1 ;;
    *) ARGS+=("${arg}") ;;
  esac
done

mkdir -p "${LOG_DIR}"

# Build agent list: each item "agent_name|config_abs_path"
AGENTS=()
if [ ${#ARGS[@]} -eq 0 ]; then
  # Defaults if none provided
  AGENTS+=("go1_0|${REPO_ROOT}/sim2real/config/go1_0.yaml")
  AGENTS+=("go1_1|${REPO_ROOT}/sim2real/config/go1_1.yaml")
else
  for item in "${ARGS[@]}"; do
    if [[ "${item}" == *:* ]]; then
      agent="${item%%:*}"
      cfg="${item#*:}"
      # Make config absolute if not already
      if [[ "${cfg}" != /* ]]; then
        cfg="${REPO_ROOT}/${cfg}"
      fi
      AGENTS+=("${agent}|${cfg}")
    else
      agent="${item}"
      cfg="${REPO_ROOT}/sim2real/config/${agent}.yaml"
      AGENTS+=("${agent}|${cfg}")
    fi
  done
fi

echo "Repo root: ${REPO_ROOT}"
echo "Using python: ${PYTHON_BIN}"
echo "Log dir: ${LOG_DIR}"
echo "Agents to launch:"
for a in "${AGENTS[@]}"; do
  agent="${a%%|*}"; cfg="${a#*|}"
  echo "  - ${agent} with ${cfg}"
done

# Ensure the module can be found
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

PIDS=()

start_agent () {
  local agent_name="$1"
  local config_path="$2"
  local log_file="$3"

  if [ ! -f "${config_path}" ]; then
    echo "ERROR: Config not found for ${agent_name}: ${config_path}" >&2
    exit 1
  fi

  echo "Starting ${agent_name} (config: ${config_path}) → log: ${log_file}"
  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[DRY-RUN] ${PYTHON_BIN} sim2real/utils/robot_interface/go1_interface.py --agent_name \"${agent_name}\" --config \"${config_path}\" >\"${log_file}\" 2>&1 &"
    return
  fi

  "${PYTHON_BIN}" sim2real/utils/robot_interface/go1_interface.py \
    --agent_name "${agent_name}" \
    --config "${config_path}" \
    >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

for a in "${AGENTS[@]}"; do
  agent="${a%%|*}"; cfg="${a#*|}"
  start_agent "${agent}" "${cfg}" "${LOG_DIR}/${agent}.log"
done

cleanup() {
  if [ "${DRY_RUN}" -eq 1 ]; then
    return
  fi
  if [ "${#PIDS[@]}" -gt 0 ]; then
    echo "Stopping agents: ${PIDS[*]}"
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
  fi
}

trap cleanup EXIT INT TERM

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "Dry-run complete. No processes started."
  exit 0
fi

if [ "${NO_TAIL}" -eq 1 ]; then
  echo "Agents started. Not tailing logs (--no-tail set)."
  for a in "${AGENTS[@]}"; do
    agent="${a%%|*}"
    echo "  - ${LOG_DIR}/${agent}.log"
  done
  # Wait to keep trap active until killed
  wait
else
  echo "Agents started. Tailing logs (Ctrl+C to stop and cleanup)."
  for a in "${AGENTS[@]}"; do
    agent="${a%%|*}"
    echo "  - ${LOG_DIR}/${agent}.log"
  done
  # tail -F all logs
  tails=()
  for a in "${AGENTS[@]}"; do
    agent="${a%%|*}"
    tails+=("-F" "${LOG_DIR}/${agent}.log")
  done
  tail "${tails[@]}"
fi