#!/bin/bash
# Download the sherpa-onnx streaming Zipformer bilingual (Chinese + English) ASR model
# that smart_stage_brain.py expects at
#   ~/smart-stage/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
#
# ~260 MB compressed. Source: https://github.com/k2-fsa/sherpa-onnx/releases
#
# Run from the repo root:
#   bash models/download.sh
set -euo pipefail

MODEL_NAME="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2"
DEST_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${DEST_DIR}/${MODEL_NAME}"

if [[ -d "${TARGET}" ]]; then
  echo "Model already present at ${TARGET}"
  exit 0
fi

echo "Downloading ${MODEL_NAME}.tar.bz2 ..."
curl -L -o "${DEST_DIR}/${MODEL_NAME}.tar.bz2" "${MODEL_URL}"

echo "Extracting ..."
tar -xjf "${DEST_DIR}/${MODEL_NAME}.tar.bz2" -C "${DEST_DIR}"
rm "${DEST_DIR}/${MODEL_NAME}.tar.bz2"

echo "Done. Model at ${TARGET}"
echo ""
echo "If your smart-stage deployment lives at ~/smart-stage/, symlink or copy:"
echo "  ln -s \"${TARGET}\" ~/smart-stage/models/${MODEL_NAME}"
