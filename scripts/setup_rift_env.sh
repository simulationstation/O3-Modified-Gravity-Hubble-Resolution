#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT}/.venv_rift"
RIFT_VERSION="${RIFT_VERSION:-0.0.17.7}"

if [[ ! -d "${VENV_PATH}" ]]; then
  python3 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "RIFT==${RIFT_VERSION}"

echo "[ok] venv: ${VENV_PATH}"
echo "[ok] RIFT version: ${RIFT_VERSION}"
echo "[ok] util_RIFT_pseudo_pipe.py: $(command -v util_RIFT_pseudo_pipe.py)"
echo "[ok] integrate_likelihood_extrinsic_batchmode: $(command -v integrate_likelihood_extrinsic_batchmode)"
