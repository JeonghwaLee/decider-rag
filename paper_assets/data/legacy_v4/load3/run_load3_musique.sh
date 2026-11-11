#!/bin/bash
cd .; R=runs; PY=python; O=$R/load3
export HF_HUB_OFFLINE=1 HF_HOME=$HOME/.cache/huggingface
until grep -q "2wiki test2 judging done" $R/judge_final/driver.log; do sleep 30; done
for g in 2 3 6; do until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g) -lt 2000 ]; do sleep 10; done; done
CUDA_VISIBLE_DEVICES=2 .venv/bin/vllm serve models/Qwen3.8-Flash-Next-FP8 --served-model-name q38-flash-next-fp8 --port 8393 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 64 > $O/logs/vllm_cloud_8393.log 2>&1 &
echo "$(date -u +%FT%TZ) set B cloud PID $! :8393 GPU2" >> $O/driver.log
CUDA_VISIBLE_DEVICES=3 .venv/bin/vllm serve Qwen/Qwen3-4B --served-model-name qwen3-4b --port 8395 --max-model-len 16384 --gpu-memory-utilization 0.5 --max-num-seqs 128 > $O/logs/vllm_edge_8395.log 2>&1 &
echo "$(date -u +%FT%TZ) set B edge PID $! :8395 GPU3" >> $O/driver.log
CUDA_VISIBLE_DEVICES=6 .venv/bin/vllm serve $R/llmsel/vllm_dec_joint --served-model-name dec-joint --port 8398 --runner pooling --convert embed --pooler-config '{"pooling_type": "LAST", "use_activation": false}' --max-model-len 8192 --gpu-memory-utilization 0.2 > $O/logs/vllm_dec_joint_8398.log 2>&1 &
echo "$(date -u +%FT%TZ) set B decider PID $! :8398 GPU6" >> $O/driver.log
until [ $(cat $O/logs/vllm_cloud_8393.log $O/logs/vllm_edge_8395.log $O/logs/vllm_dec_joint_8398.log | grep -c "startup complete") -ge 3 ]; do sleep 15; done
J=$R/pilot9/musique_test_fbN.json
g() { python3 -c "import json;v=json.load(open('$J'))['thr']['$1'];print(':'.join(str(x) for x in v))"; }
TM=$R/e2e4/musique_llm_thresholds.json
AK=$(python3 -c "import json;v=json.load(open('$TM'))['think'];print(f\"{v['a']}:{v['b']}\")"); AM=$(python3 -c "import json;v=json.load(open('$TM'))['sh'];print(f\"{v['a']}:{v['b']}\")")
JT=$(g "joint_s1234|E/N/T|x1.0")
MPOL="joint_vllm=vdec:8398:dec-joint:$R/llmsel/vllm_dec_joint:$JT qwen38_think=llm:think:edge+think:$AK qwen38_shared=llm:shared:edge+think:$AM tfidf_x1=tfidf:$(g "tfidf|E/N/T|x1.0") ruleT_tuned=ruleT_tuned:$(g "ruleT_tuned|-|x1.0")"
cap() { python3 -c "
import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; ms=max(x['t_done'] for x in r); c=len(r)/ms
print(' '.join(f'{f*c:.3g}' for f in (0.3,0.8,1.2)))" $1; }
MCOM="--ds musique --idx $R/pilot9/trace/_index --tfidf_train pilot9:musique:train --fallback N --workers 4 --cloud http://127.0.0.1:8393/v1 --edge http://127.0.0.1:8395/v1"
CUDA_VISIBLE_DEVICES=6 $PY src/ecrag/load_ent3.py $MCOM --runs train/pool_00 --pol ${MPOL##* } --burst 200 --out $O/musique_calib > $O/logs/musique_calib.log 2>&1
MR=$(cap $O/musique_calib/burst_ruleT_tuned_n200.jsonl); echo "$(date -u +%FT%TZ) musique calib (set B) rc=$? rates $MR" >> $O/driver.log
CUDA_VISIBLE_DEVICES=6 $PY src/ecrag/load_ent3.py $MCOM --runs test/pool_00 test/pool_01 --pol $MPOL --rates $MR --seeds 1 2 --out $O/musique_test > $O/logs/musique_test.log 2>&1
echo "$(date -u +%FT%TZ) musique test (set B) rc=$?" >> $O/driver.log
