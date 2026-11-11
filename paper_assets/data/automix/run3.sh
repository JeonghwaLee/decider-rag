#!/bin/bash
cd .; R=runs; O=$R/automix; PY=python
export HF_HUB_OFFLINE=1 HF_HOME=$HOME/.cache/huggingface
log() { echo "$(date -u +%FT%TZ) [run3] $*" >> $O/driver.log; }
AB=$(python3 -c "import json;print(json.load(open('$R/pilot8/en_eval_big2.json'))['thresholds']['1.0']['big2e4'])")
J=$R/pilot9/musique_test_fbN.json; JT=$(python3 -c "import json;v=json.load(open('$J'))['thr']['joint_s1234|E/N/T|x1.0'];print(':'.join(str(x) for x in v))")
T4=$(python3 -c "import json;print(json.load(open('$O/thr_hotpot_k4.json'))['thr'])"); T1=$(python3 -c "import json;print(json.load(open('$O/thr_musique_k1.json'))['thr'])")
HPOL="big2e4_vllm=vdec:8197:dec-big2e4:$R/llmsel/vllm_dec_big2e4:$AB:inf automix_k4=automix:E:$T4:4"
MPOL="joint_vllm=vdec:8198:dec-joint:$R/llmsel/vllm_dec_joint:$JT automix_k1=automix:N:$T1:1"
MCOM="--ds musique --idx $R/pilot9/trace/_index --trace $R/pilot9/trace/musique --fallback N --block 25"
HCOM="--ds hotpotqa --idx $R/pilot8/trace/_index --trace $R/pilot8/trace/hotpotqa --fallback E --block 25"
log "thresholds k4 hotpot $T4, k1 musique $T1"
( until [ $(cat $O/musique_test_b/*_r0.jsonl 2>/dev/null | wc -l) -ge 800 ]; do sleep 60; done; sleep 30
  CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/e2e_ent6.py $MCOM --cloud_port 8393 --edge_port 8395 --runs test/pool_00 --limit 1 --out $O/smoke_k1 --pol $MPOL > $O/logs/smoke_k1.log 2>&1 || { log "smoke k1 failed"; exit 1; }
  log "musique k1 live start (set B)"
  CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/e2e_ent6.py $MCOM --cloud_port 8393 --edge_port 8395 --runs test/pool_00 test/pool_01 --out $O/musique_k1 --pol $MPOL > $O/logs/musique_k1.log 2>&1
  log "musique k1 rc=$?" ) &
( until grep -q 'musique pool_01 rc=' $O/driver.log; do sleep 60; done
  CUDA_VISIBLE_DEVICES=2 $PY src/ecrag/e2e_ent6.py $HCOM --cloud_port 8193 --edge_port 8195 --runs test6/pool_00 --limit 1 --out $O/smoke_k4 --pol $HPOL > $O/logs/smoke_k4.log 2>&1 || { log "smoke k4 failed"; exit 1; }
  log "hotpot k4 live start (set A)"
  CUDA_VISIBLE_DEVICES=2 $PY src/ecrag/e2e_ent6.py $HCOM --cloud_port 8193 --edge_port 8195 --runs test6/pool_00 test6/pool_01 test6/pool_02 test6/pool_03 test6/pool_04 test6/pool_05 test6/pool_06 test6/pool_07 --out $O/hotpot_k4 --pol $HPOL > $O/logs/hotpot_k4.log 2>&1
  log "hotpot k4 rc=$?" ) &
wait; log "run3 done"
