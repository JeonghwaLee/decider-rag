# Decider-RAG

Code and recorded results for **Decider-RAG: Learning to Plan Before Generating in Edge–Cloud
Retrieval-Augmented Generation** (ICDE 2027 submission).

A 4B value model (the *Decider*) reads the question, the pruned evidence and the number of retained
passages once, predicts the quality gain of each execution plan, and a validation-calibrated policy
chooses one plan before any answer is generated.

| Paper | Code | Plan |
|---|---|---|
| E | `E` | edge answer (Qwen3-4B) |
| C | `N` | direct cloud answer (thinking off) |
| R | `T` | cloud reasoning answer (thinking on) |

The code uses the original letters `E/N/T`; the paper writes them as `E/C/R`.

## Layout

```
src/ecrag/        experiment code (data building, traces, training, calibration, baselines,
                  complete-pipeline runs, load tests, judges, analysis)
src/ecrag/paths.py  all filesystem locations (override with environment variables)
paper_assets/     script, recorded per-request logs and tables used to build the paper's
                  tables and figures
requirements.txt
```

## Reproducing the paper tables and figures

The recorded outputs needed for the new complete-pipeline campaign are in `paper_assets/data/`.

```bash
cd paper_assets
python build_assets.py
```

This rebuilds Tables II, III and V (`tables/pipeline.tex`, `tables/samples.tex`,
`tables/legacy_baselines.tex`), Figures 2 and 3 (`figures/fig01_tradeoffs.pdf`,
`figures/fig03_stages.pdf`) and `data/metrics_v5.json`, after checking every input file against
`data/manifest.json`; paired bootstrap intervals are recomputed from the request logs.

- `paper_assets/data/automix/`: complete-pipeline logs of the Decider, the AutoMix-style cascade
  (s = 8, 4, 1) and the Qwen3.8 selectors (Tables II and III, Figures 2 and 3).
- `paper_assets/data/p0/`: Decider versus Decider-Post (Table IV): live logs, frozen validation
  policies (`*/evaluation/PRIMARY.json`), budget-controlled replay (`*/evaluation/REPLAY.json`) and
  the live summary (`FINAL_SUMMARY.md`, `LIVE_SUMMARY.json`). `build_assets.py` does not rebuild
  Table IV; regenerate its summary with

  ```bash
  cd src/ecrag
  python summarize_postgen.py --root ../../paper_assets/data/p0
  ```

  The reference deadline D is read from the non-test calibration runs in `paper_assets/data/e2e4/`.
- `paper_assets/data/legacy_v4/`: recorded summaries of the earlier campaign (baselines,
  equal-spend selection, load, judges, ablations; Tables V–X). Tables VI and VIII–X are provided as
  built in `paper_assets/tables/legacy/`; Table VII is `paper_assets/tables/latency_v5.tex` (same
  numbers as `tables/legacy/latency.tex`, with a clearer header).

## Running the experiments

1. Place the EC-RAG implementation (Wang et al., ICDE 2026; commit `1379d14`) at
   `third_party/ec_rag/` or set `ECRAG_SRC`. It is used for indexing, retrieval, reranking and
   pruning and is not redistributed here.
2. Datasets come from the Hugging Face hub (HotpotQA, MuSiQue, 2WikiMultihopQA, LongBench);
   `build_data.py`, `build_musique.py`, `build_2wiki.py` and `build_test*.py` create the pools and
   splits.
3. Serve the generators with vLLM: edge `Qwen/Qwen3-4B`, cloud `Qwen3.8-Flash-Next-FP8`
   (`CLOUD_MODEL_PATH`). Retrieval uses `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-v2-m3`.
4. Run from the repository root; outputs go to `runs/` (`DECIDER_RAG_RUNS`).

Main entry points:

| Step | Scripts |
|---|---|
| Retrieval traces and stored answers of every plan | `stage1_trace.py`, `regen_answers.py`, `stage2_cloud.py` |
| Decider training (Qwen3-4B + LoRA + five-output head) | `train_router4.py`, `train_router5.py` |
| Serving the Decider with vLLM (pooling) | `export_decider_vllm.py`, `vllm_decider.py` |
| Validation calibration and frozen policies | `val_thresholds.py`, `freeze_ent_policy.py` |
| LLM selectors, TF-IDF, EC-RAG rule | `llm_selector.py`, `llm_selector_x.py`, `llm_selector_shared.py`, `analyze_ent2.py`, `rule_tuned.py` |
| Equal-spend replay | `online_cmp.py` |
| Complete-pipeline runs | `e2e_ent6.py` |
| AutoMix-style cascade | `automix_verify.py`, `automix_thr.py`, `automix_cmp.py`, `automix_k.py` |
| Decider-Post | `train_learned_postgen.py`, `predict_postgen_vllm.py`, `evaluate_learned_postgen.py`, `e2e_postgen.py`, `summarize_postgen.py` |
| Open-loop load | `load_ent3.py` |
| Public LLM judges | `judge_final.py`, `judge_rubric.py` |
| Paired bootstrap | `bootlib.py` |

## Checkpoints

The five trained models used in the paper (Decider for HotpotQA, MuSiQue and 2Wiki; Decider-Post
for HotpotQA and MuSiQue) are released separately as LoRA + head checkpoints (about 51 MB each)
with their validation-calibrated policies, a standalone loader (`decider.py`) and a smoke test
that reproduces the recorded scores:
<https://huggingface.co/norip76/decider-rag-checkpoints>.

Full retrieval traces and stored plan answers (HotpotQA 5.1 GB, MuSiQue 1.8 GB, 2Wiki 1.3 GB),
needed to re-run training and the offline replays end to end, are not part of this repository.

## License

The code in this repository is released under the MIT License (see `LICENSE`). Third-party
datasets, models and the EC-RAG implementation keep their own licenses.
