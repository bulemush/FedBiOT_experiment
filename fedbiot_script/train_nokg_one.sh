#!/usr/bin/env bash
set -euo pipefail

METHOD="${METHOD:-${1:-}}"
DATASET="${DATASET:-${2:-}}"
SEEDS="${SEEDS:-1 2 3}"
GPU="${GPU:-0,1}"
LOGDIR="${LOGDIR:-logs/nokg}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-}"
LOCAL_STEPS="${LOCAL_STEPS:-}"
ALIGN_STEPS="${ALIGN_STEPS:-}"

if [[ -z "${METHOD}" || -z "${DATASET}" ]]; then
  echo "Usage: METHOD=fedbiot DATASET=cwq bash fedbiot_script/train_nokg_one.sh"
  echo "   or: bash fedbiot_script/train_nokg_one.sh fedbiot cwq"
  exit 1
fi

case "${METHOD}" in
  fedbiot|fedot) ;;
  *)
    echo "METHOD must be one of: fedbiot, fedot"
    exit 1
    ;;
esac

case "${DATASET}" in
  cwq|graphquestions|kqapro|openbookqa) ;;
  *)
    echo "DATASET must be one of: cwq, graphquestions, kqapro, openbookqa"
    exit 1
    ;;
esac

cfg="fedbiot_script/${METHOD}_nokg/${DATASET}.yaml"
if [[ ! -f "${cfg}" ]]; then
  echo "Missing config: ${cfg}"
  exit 1
fi

mkdir -p "${LOGDIR}" "checkpoints/nokg/${METHOD}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for seed in ${SEEDS}; do
  save_to="checkpoints/nokg/${METHOD}/${DATASET}_seed${seed}.ckpt"
  expname="${METHOD}_nokg/${DATASET}_seed${seed}"
  log="${LOGDIR}/train_${METHOD}_${DATASET}_seed${seed}.log"

  extra_args=()
  if [[ -n "${TOTAL_ROUNDS}" ]]; then
    extra_args+=(federate.total_round_num "${TOTAL_ROUNDS}")
  fi
  if [[ -n "${LOCAL_STEPS}" ]]; then
    extra_args+=(train.local_update_steps "${LOCAL_STEPS}")
  fi
  if [[ -n "${ALIGN_STEPS}" ]]; then
    extra_args+=(llm.offsite_tuning.emu_align.train.local_update_steps "${ALIGN_STEPS}")
  fi

  echo "[START] ${METHOD} ${DATASET} seed=${seed}"
  python federatedscope/main.py --cfg "${cfg}" \
    seed "${seed}" \
    device 0 \
    dataloader.batch_size "${BATCH_SIZE}" \
    federate.save_to "${save_to}" \
    expname "${expname}" \
    "${extra_args[@]}" \
    > "${log}" 2>&1
  echo "[DONE] ${METHOD} ${DATASET} seed=${seed}"
done
