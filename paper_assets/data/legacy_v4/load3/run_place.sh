#!/bin/bash
cd .; R=runs; PY=python; O=$R/load3
export HF_HUB_OFFLINE=1 HF_HOME=$HOME/.cache/huggingface
log() { echo "$(date -u +%FT%TZ) $*" >> $O/driver.log; }
until grep -q "musique test (set B) rc" $O/driver.log; do sleep 60; done
for pp in 1418890:8393 1418892:8395 1418894:8398; do
  pid=${pp%%:*}; port=${pp##*:}
  if ps -p $pid -o args= | grep -q -- "--port $port"; then kill $pid; log "set B PID $pid (:$port) stopped"; else log "set B PID $pid not matching :$port, left alone"; fi
done
sleep 60
CUDA_VISIBLE_DEVICES=6 nohup .venv/bin/vllm serve $R/llmsel/vllm_dec_big2e4 --served-model-name dec-big2e4 --port 8297 --runner pooling --convert embed \
  --pooler-config '{"pooling_type": "LAST", "use_activation": false}' --max-model-len 8192 --gpu-memory-utilization 0.2 > $O/logs/dec_sep_8297.log 2>&1 &
DP=$!; log "placement: separate decider PID $DP :8297 GPU6"
until curl -s localhost:8297/health >/dev/null; do sleep 10; done
AB=$(python3 -c "import json;print(json.load(open('$R/pilot8/en_eval_big2.json'))['thresholds']['1.0']['big2e4'])")
POL="big2e4_shared=vdec:8197:dec-big2e4:$R/llmsel/vllm_dec_big2e4:$AB:inf big2e4_sep=vdec:8297:dec-big2e4:$R/llmsel/vllm_dec_big2e4:$AB:inf"
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/load_ent3.py --ds hotpotqa --idx $R/pilot8/trace/_index --tfidf_train legacy:train --fallback E --workers 4 \
  --runs test6/pool_00 test6/pool_01 test6/pool_02 test6/pool_03 --pol $POL --rates 6.4 9.6 --seeds 1 2 --out $O/hotpot_place > $O/logs/hotpot_place.log 2>&1
log "placement hotpot rc=$?"
