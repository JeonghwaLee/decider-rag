#!/bin/bash
cd .; R=runs; PY=python; O=$R/load3
export HF_HUB_OFFLINE=1 HF_HOME=$HOME/.cache/huggingface
A=$(python3 -c "import json;print(json.load(open('$R/pilot7/ent_policy.json'))['policies']['tfidf|E/N|x1.0']['a_dEN'])")
AB=$(python3 -c "import json;print(json.load(open('$R/pilot8/en_eval_big2.json'))['thresholds']['1.0']['big2e4'])")
RN=$(python3 -c "import json;v=json.load(open('$R/pilot7/rule_tuned.json'))['policies']['ruleN_tuned|x1.0'];print(f\"{v['lam']}:{v['c']}\")")
TH=$R/e2e4/hotpot_llm_thresholds.json
AZ=$(python3 -c "import json;print(json.load(open('$TH'))['zs']['a'])"); AS=$(python3 -c "import json;print(json.load(open('$TH'))['sh']['a'])")
HPOL="big2e4_vllm=vdec:8197:dec-big2e4:$R/llmsel/vllm_dec_big2e4:$AB:inf qwen38_zs=llm:zs:edge:$AZ:inf qwen38_shared=llm:shared:edge:$AS:inf tfidf_EN=tfidf:$A:inf ruleN_tuned=ruleN_tuned:$RN"
J=$R/pilot9/musique_test_fbN.json
g() { python3 -c "import json;v=json.load(open('$J'))['thr']['$1'];print(':'.join(str(x) for x in v))"; }
TM=$R/e2e4/musique_llm_thresholds.json
AK=$(python3 -c "import json;v=json.load(open('$TM'))['think'];print(f\"{v['a']}:{v['b']}\")"); AM=$(python3 -c "import json;v=json.load(open('$TM'))['sh'];print(f\"{v['a']}:{v['b']}\")")
JT=$(g "joint_s1234|E/N/T|x1.0")
MPOL="joint_vllm=vdec:8198:dec-joint:$R/llmsel/vllm_dec_joint:$JT qwen38_think=llm:think:edge+think:$AK qwen38_shared=llm:shared:edge+think:$AM tfidf_x1=tfidf:$(g "tfidf|E/N/T|x1.0") ruleT_tuned=ruleT_tuned:$(g "ruleT_tuned|-|x1.0")"
cap() { python3 -c "
import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; ms=max(x['t_done'] for x in r); c=len(r)/ms
print(' '.join(f'{f*c:.3g}' for f in (0.3,0.8,1.2)))" $1; }
HCOM="--ds hotpotqa --idx $R/pilot8/trace/_index --tfidf_train legacy:train --fallback E --workers 4"
MCOM="--ds musique --idx $R/pilot9/trace/_index --tfidf_train pilot9:musique:train --fallback N --workers 4"
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/load_ent3.py $HCOM --runs train2/pool_00 --pol ${HPOL##* } --burst 200 --out $O/hotpot_calib > $O/logs/hotpot_calib.log 2>&1
HR=$(cap $O/hotpot_calib/burst_ruleN_tuned_n200.jsonl); echo "$(date -u +%FT%TZ) hotpot calib rc=$? rates $HR" >> $O/driver.log
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/load_ent3.py $HCOM --runs test6/pool_00 test6/pool_01 test6/pool_02 test6/pool_03 --pol $HPOL --rates $HR --seeds 1 2 --out $O/hotpot_test6 > $O/logs/hotpot_test6.log 2>&1
echo "$(date -u +%FT%TZ) hotpot test6 rc=$?" >> $O/driver.log
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/load_ent3.py $MCOM --runs train/pool_00 --pol ${MPOL##* } --burst 200 --out $O/musique_calib > $O/logs/musique_calib.log 2>&1
MR=$(cap $O/musique_calib/burst_ruleT_tuned_n200.jsonl); echo "$(date -u +%FT%TZ) musique calib rc=$? rates $MR" >> $O/driver.log
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/load_ent3.py $MCOM --runs test/pool_00 test/pool_01 --pol $MPOL --rates $MR --seeds 1 2 --out $O/musique_test > $O/logs/musique_test.log 2>&1
echo "$(date -u +%FT%TZ) musique test rc=$?" >> $O/driver.log
