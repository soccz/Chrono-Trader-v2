#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv_local/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv_local/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "Python interpreter not found" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

compile_targets=(
  app.py
  main.py
  utils/command_modes.py
  utils/legacy_modes.py
  utils/scheduled_modes.py
  utils/output_contract.py
  utils/ops_health.py
  scripts/run_scheduled.py
  scripts/run_ops_job.py
  scripts/ops_doctor.py
  scripts/backfill_output_contracts.py
  scripts/ops_validate.py
  scripts/validate_ops_contracts.py
  scripts/ops_healthcheck.py
)

quick_tests=(
  tests/test_ops_api.py
  tests/test_ops_history.py
  tests/test_system_resources.py
  tests/test_task_runner.py
  tests/test_backfill_output_contracts.py
  tests/test_output_contract.py
  tests/test_ops_health.py
  tests/test_ops_doctor.py
  tests/test_run_ops_job.py
  tests/test_run_scheduled.py
  tests/test_bucket_quota_parse.py
  tests/test_freshness.py
  tests/test_market_selection.py
  tests/test_minrec_policy.py
  tests/test_minrec_synthetic_fallback.py
  tests/test_price_cache.py
  tests/test_run_lock.py
)

quick_tests_ml=(
  tests/test_recommender_filters.py
  tests/test_recommender_price_fallback.py
)

fallback_unittest_scripts=(
  tests/test_price_cache.py
)

run_compile() {
  echo "[verify] python compile check"
  "$PYTHON" -m py_compile "${compile_targets[@]}"
}

run_contract_checks() {
  echo "[verify] ops contract validation"
  "$PYTHON" scripts/validate_ops_contracts.py --autofix-legacy
}

run_ops_doctor() {
  echo "[verify] ops doctor"
  "$PYTHON" scripts/ops_doctor.py --json >/dev/null
}

ensure_runtime_deps() {
  if "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
needed = ("pandas", "numpy")
missing = [name for name in needed if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
  then
    return
  fi

  echo "[verify] missing runtime dependencies on selected interpreter: pandas/numpy" >&2
  exit 4
}

has_pytest() {
  "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("pytest") else 1)
PY
}

has_torch() {
  "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
}

run_quick_tests() {
  if has_pytest; then
    echo "[verify] quick pytest suite"
    "$PYTHON" -m pytest -q "${quick_tests[@]}"
    if has_torch; then
      echo "[verify] quick pytest ML suite"
      "$PYTHON" -m pytest -q "${quick_tests_ml[@]}"
    else
      echo "[verify] skipping ML pytest subset because torch is unavailable" >&2
    fi
    return
  fi

  ensure_runtime_deps
  echo "[verify] pytest not available, falling back to unittest core suite"
  for test_script in "${fallback_unittest_scripts[@]}"; do
    "$PYTHON" "$test_script" -q
  done
}

run_full_tests() {
  if has_pytest; then
    echo "[verify] full pytest suite"
    "$PYTHON" -m pytest -q tests
    return
  fi

  ensure_runtime_deps
  echo "[verify] pytest not available, running unittest fallback only" >&2
  for test_script in "${fallback_unittest_scripts[@]}"; do
    "$PYTHON" "$test_script" -q
  done
  echo "[verify] full suite incomplete because pytest is missing" >&2
}

case "$MODE" in
  compile)
    run_compile
    ;;
  quick)
    run_compile
    run_contract_checks
    run_ops_doctor
    run_quick_tests
    ;;
  full)
    run_compile
    run_contract_checks
    run_ops_doctor
    run_full_tests
    ;;
  *)
    echo "Usage: scripts/verify_site.sh [compile|quick|full]" >&2
    exit 2
    ;;
esac
