#!/bin/bash
cd .; R=runs; O=$R/automix; PY=python; VP=.venv/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=$HOME/.cache/huggingface
mkdir -p $O/logs $O/ver; log() { echo "$(date -u +%FT%TZ) $*" >> $O/driver.log; }
U=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1<=5{print $2}' | paste -sd'|')
log "run2: servers from run.sh already up (PIDs in logs/pid_*); verifier: edge raw, cloud chat"
for p in 8193 8195 8197 8198 8393 8395; do curl -s -m 3 localhost:$p/health >/dev/null || { log "port $p down - abort"; exit 1; }; done

$VP src/ecrag/automix_verify.py --spec legacy:val --cand E --port 8195 --model qwen3-4b --out $O/ver/legacy_val.jsonl > $O/logs/ver_h_val.log 2>&1 &
$VP src/ecrag/automix_verify.py --spec pilot8:hotpotqa:test4 --cand E --port 8395 --model qwen3-4b --out $O/ver/pilot8_hotpotqa_test4.jsonl > $O/logs/ver_h_t4.log 2>&1 &
$VP src/ecrag/automix_verify.py --spec pilot9:musique:val --cand N --port 8193 --model q38-flash-next-fp8 --out $O/ver/pilot9_musique_val.jsonl > $O/logs/ver_m_val.log 2>&1 &
$VP src/ecrag/automix_verify.py --spec pilot9:musique:test --cand N --port 8393 --model q38-flash-next-fp8 --out $O/ver/pilot9_musique_test.jsonl > $O/logs/ver_m_test.log 2>&1 &
wait
$VP src/ecrag/automix_verify.py --spec pilot8:hotpotqa:test5 --cand E --port 8195 --model qwen3-4b --out $O/ver/pilot8_hotpotqa_test5.jsonl > $O/logs/ver_h_t5.log 2>&1 &
$VP src/ecrag/automix_verify.py --spec pilot8:hotpotqa:test6 --cand E --port 8395 --model qwen3-4b --out $O/ver/pilot8_hotpotqa_test6.jsonl > $O/logs/ver_h_t6.log 2>&1 &
wait
NV=$(cat $O/ver/*.jsonl | wc -l); log "offline verification done: $NV rows (expected 4800)"
[ "$NV" -eq 4800 ] || { log "verification rows mismatch - abort before live runs"; exit 1; }

$VP src/ecrag/automix_thr.py --val legacy:val --fallback E --cand E --ver_dir $O/ver --out $O/thr_hotpot.json > $O/logs/thr.log 2>&1
$VP src/ecrag/automix_thr.py --val pilot9:musique:val --fallback N --cand N --ver_dir $O/ver --out $O/thr_musique.json >> $O/logs/thr.log 2>&1
TA=$(python3 -c "import json;print(json.load(open('$O/thr_hotpot.json'))['thr'])"); TM=$(python3 -c "import json;print(json.load(open('$O/thr_musique.json'))['thr'])")
log "thresholds hotpot $TA musique $TM"

H='pred_{split}.json'; M='pred_pilot9_musique_{split}.json'; S=$R/llmsel
( $VP src/ecrag/automix_cmp.py --val legacy:val --test pilot8:hotpotqa:test4 pilot8:hotpotqa:test5 pilot8:hotpotqa:test6 --fallback E --fam E/N --cand E --ver_dir $O/ver \
    --scorers decider=$S/vllm_pred/big2e4/$H qwen38_zs=$S/qwen38/two/$H qwen38_shared=$S/qwen38_shared/two/$H glm=$S/glm53/two/$H qwen38_think=$S/qwen38_think/two/$H \
    --mults 0.5 0.75 1.0 1.5 --out $O/offline_hotpot.txt > $O/logs/offline_hotpot.log 2>&1
  $VP src/ecrag/automix_cmp.py --val pilot9:musique:val --test pilot9:musique:test --fallback N --fam E/N/T --cand N --ver_dir $O/ver \
    --scorers decider=$S/vllm_pred/joint/$M qwen38_think=$S/qwen38_think/two/$M qwen38_shared=$S/qwen38_shared/two/$M glm=$S/glm53/choice/$M qwen38_zs=$S/qwen38/two/$M \
    --mults 0.5 0.75 1.0 1.5 --out $O/offline_musique.txt > $O/logs/offline_musique.log 2>&1; log "offline comparison done" ) &

AB=$(python3 -c "import json;print(json.load(open('$R/pilot8/en_eval_big2.json'))['thresholds']['1.0']['big2e4'])")
TH=$R/e2e4/hotpot_llm_thresholds.json
AZ=$(python3 -c "import json;print(json.load(open('$TH'))['zs']['a'])"); AS=$(python3 -c "import json;print(json.load(open('$TH'))['sh']['a'])")
J=$R/pilot9/musique_test_fbN.json; JT=$(python3 -c "import json;v=json.load(open('$J'))['thr']['joint_s1234|E/N/T|x1.0'];print(':'.join(str(x) for x in v))")
TMJ=$R/e2e4/musique_llm_thresholds.json
AK=$(python3 -c "import json;v=json.load(open('$TMJ'))['think'];print(f\"{v['a']}:{v['b']}\")"); AM=$(python3 -c "import json;v=json.load(open('$TMJ'))['sh'];print(f\"{v['a']}:{v['b']}\")")
HPOL="big2e4_vllm=vdec:8197:dec-big2e4:$R/llmsel/vllm_dec_big2e4:$AB:inf automix=automix:E:$TA qwen38_zs=llm:zs:edge:$AZ:inf qwen38_shared=llm:shared:edge:$AS:inf"
MPOL="joint_vllm=vdec:8198:dec-joint:$R/llmsel/vllm_dec_joint:$JT automix=automix:N:$TM qwen38_think=llm:think:edge+think:$AK qwen38_shared=llm:shared:edge+think:$AM"
MCOM="--ds musique --idx $R/pilot9/trace/_index --trace $R/pilot9/trace/musique --fallback N --block 25"
CUDA_VISIBLE_DEVICES=2 $PY src/ecrag/e2e_ent6.py --ds hotpotqa --idx $R/pilot8/trace/_index --trace $R/pilot8/trace/hotpotqa --fallback E --block 25 \
  --cloud_port 8193 --edge_port 8195 --runs test6/pool_00 --limit 2 --out $O/smoke_hotpot --pol $HPOL > $O/logs/smoke_hotpot.log 2>&1 || { log "smoke hotpot failed"; exit 1; }
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/e2e_ent6.py $MCOM --cloud_port 8393 --edge_port 8395 --runs test/pool_00 --limit 2 --out $O/smoke_musique --pol $MPOL > $O/logs/smoke_musique.log 2>&1 || { log "smoke musique failed"; exit 1; }
log "smoke ok: $(cat $O/smoke_*/*_r0.jsonl | wc -l) rows"
log "live start: hotpot (set A) + musique pool_00 (set B)"
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/e2e_ent6.py $MCOM --cloud_port 8393 --edge_port 8395 --runs test/pool_00 --out $O/musique_test_b --pol $MPOL > $O/logs/musique_b.log 2>&1 &
PB=$!
CUDA_VISIBLE_DEVICES=2 $PY src/ecrag/e2e_ent6.py --ds hotpotqa --idx $R/pilot8/trace/_index --trace $R/pilot8/trace/hotpotqa --fallback E --block 25 \
  --cloud_port 8193 --edge_port 8195 --runs test6/pool_00 test6/pool_01 test6/pool_02 test6/pool_03 test6/pool_04 test6/pool_05 test6/pool_06 test6/pool_07 \
  --out $O/hotpot_test6 --pol $HPOL > $O/logs/hotpot.log 2>&1
log "hotpot live rc=$?; musique pool_01 on set A"
CUDA_VISIBLE_DEVICES=2 $PY src/ecrag/e2e_ent6.py $MCOM --cloud_port 8193 --edge_port 8195 --runs test/pool_01 --out $O/musique_test_a --pol $MPOL > $O/logs/musique_a.log 2>&1
log "musique pool_01 rc=$?"
wait $PB; log "musique pool_00 rc=$?"
log "all live runs done (servers left running; stop by PID from logs/pid_*)"
