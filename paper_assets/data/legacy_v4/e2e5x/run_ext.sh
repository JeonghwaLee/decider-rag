#!/bin/bash
cd .; R=runs; PY=python; O=$R/e2e5x
export HF_HUB_OFFLINE=1 HF_HOME=$HOME/.cache/huggingface
A=$(python3 -c "import json;print(json.load(open('$R/pilot7/ent_policy.json'))['policies']['s1234_qd|E/N|x1.0']['a_dEN'])")
AT=$(python3 -c "import json;print(json.load(open('$R/pilot7/ent_policy.json'))['policies']['tfidf|E/N|x1.0']['a_dEN'])")
AB=$(python3 -c "import json;print(json.load(open('$R/pilot8/en_eval_big2.json'))['thresholds']['1.0']['big2e4'])")
RN=$(python3 -c "import json;v=json.load(open('$R/pilot7/rule_tuned.json'))['policies']['ruleN_tuned|x1.0'];print(f\"{v['lam']}:{v['c']}\")")
TH=$R/e2e4/hotpot_llm_thresholds.json
AZ=$(python3 -c "import json;print(json.load(open('$TH'))['zs']['a'])"); AS=$(python3 -c "import json;print(json.load(open('$TH'))['sh']['a'])")
BK=$R/pilot8/router/big_ent_s2345_e4
echo "$(date -u +%FT%TZ) start hotpot test6 pools 04-07" >> $O/driver.log
CUDA_VISIBLE_DEVICES=5 $PY src/ecrag/e2e_ent5.py --ds hotpotqa --idx $R/pilot8/trace/_index --trace $R/pilot8/trace/hotpotqa --tfidf_train legacy:train --fallback E \
  --block 25 --runs test6/pool_04 test6/pool_05 test6/pool_06 test6/pool_07 --out $O/hotpot_test6 \
  --pol big2e4_vllm=vdec:8197:dec-big2e4:$R/llmsel/vllm_dec_big2e4:$AB:inf big2e4_opt=decopt:$BK:$AB:inf big2e4_hf=dec:$BK:$AB:inf \
        s1234_opt=decopt:$R/pilot7/router/ent_qd_s1234:$A:inf qwen38_zs=llm:zs:edge:$AZ:inf qwen38_shared=llm:shared:edge:$AS:inf \
        tfidf_EN=tfidf:$AT:inf ruleN_tuned=ruleN_tuned:$RN allN=allN > $O/logs/hotpot_test6.log 2>&1
echo "$(date -u +%FT%TZ) hotpot test6 pools 04-07 rc=$?" >> $O/driver.log
