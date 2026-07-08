#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEEDS="${SEEDS:-1 2 3}"
LIMIT="${LIMIT:-}"
OUTDIR="${OUTDIR:-results/nokg_eval}"
LOGDIR="${LOGDIR:-logs/nokg_eval}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
METHODS="${METHODS:-fedbiot fedot}"
DATASETS="${DATASETS:-cwq graphquestions kqapro openbookqa}"

mkdir -p "${OUTDIR}" "${LOGDIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

for method in ${METHODS}; do
  for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
      cfg="fedbiot_script/${method}_nokg/${dataset}.yaml"
      ckpt="checkpoints/nokg/${method}/${dataset}_seed${seed}.ckpt"
      run_out="${OUTDIR}/${method}/${dataset}/seed${seed}"
      log="${LOGDIR}/eval_${method}_${dataset}_seed${seed}.log"

      if [[ "${SKIP_EXISTING}" == "1" && -f "${run_out}/result.json" ]]; then
        echo "[SKIP] ${method} ${dataset} seed=${seed}"
        continue
      fi

      mkdir -p "${run_out}"
      echo "[START] ${method} ${dataset} seed=${seed}"
      limit_args=()
      if [[ -n "${LIMIT}" ]]; then
        limit_args=(--limit "${LIMIT}")
      fi
      python fedbiot_script/eval_nokg_tasks.py \
        --cfg "${cfg}" \
        --dataset "${dataset}" \
        --checkpoint "${ckpt}" \
        --seed "${seed}" \
        --out "${run_out}" \
        "${limit_args[@]}" \
        > "${log}" 2>&1
      echo "[DONE] ${method} ${dataset} seed=${seed}"
    done
  done
done

python fedbiot_script/collect_nokg_eval.py --indir "${OUTDIR}"
