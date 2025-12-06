
<div align="center">

  <!-- <div>
    <img src="figs/overview.png" width="500px" alt="CureAgent"/>
  </div> -->

  <h1>CureAgent</h1>
  <em>Prize-Winning Solution (2nd Place, Track 2: Agentic Tool-Use @ CURE-Bench / NeurIPS 2025)</em>

</div>

<div align="center" style="line-height: 1; margin-bottom: 12px;">
  <br/>
  | <a href="https://jiaweizzhao.github.io/deepconf/">💻 Code</a> 
  | <a href="https://arxiv.org/abs/25xxx">📑 Tech Report (TBA)</a> 
  <!-- | <a href="figs/overview.ong">📰 Poster</a>  -->
  | <a href="https://curebench.ai/">CURE-Bench</a>
  |
</div>

## Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Supported Models](#supported-models)
- [Inference](#inference)
- [Results Snapshot](#results-snapshot)
- [Acknowledgement](#acknowledgement)
- [Contact Info](#contact-info)
- [Citation](#citation)

## Overview
We introduce an Executor-Analyst architecture that couples TxAgent (Executor) with Gemini-2.5/3 series models (Analyst) to mitigate the “context utilization failure” described in the tech report. TxAgent focuses on precise tool calls inside ToolUniverse v1.0 (211 curated biomedical APIs), while Gemini performs long-context synthesis, optional Google search, and deterministic post-processing. Our stratified late-fusion ensemble delivers competitive performance without additional training, ultimately taking 2nd place in Track 2.

<div align="center">
  <img src="figs/overview.png" width="600px" alt="Executor-Analyst Overview"/>
</div>

## Environment Setup

### 1. AMD MI300X + ROCm + vLLM (two-phase as in CURE-Bench starter)

While most experiments run on AMD MI300X, only light adjustments are needed for NVIDIA H100/V100.

```bash
# prep：clone the repo
git clone git@github.com:June01/CureAgent.git
```

```bash
# Host (Phase 1)
docker pull rocm/vllm:rocm6.3.1_vllm_0.8.5_20250513
docker run -it --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri  -p 8890:8890 -v /root/code:/app/code    aa1e9eebfc30   bash


# Container (Phase 2)
git clone git@github.com:mims-harvard/TxAgent.git
# if amd, please mv ./txagent/* TxAgent/src/txagent/
cd TxAgent                     
pip install -e .         

cd CureAgent
pip install -e .

# IMPORTANT: Use exactly tooluniverse==0.2.0 (with 211 tools) or below to avoid performance drop when tools are increased to 600+
pip install tooluniverse==0.2.0
```

### 2. GPT-OSS series 
Run the OpenAI GPT-OSS checkpoints through vLLM serving. The environment setup is following [link](https://rocm.blogs.amd.com/ecosystems-and-partners/openai-day-0/README.html).
```bash
# Bring up vLLM server (example port 8001)
python run_gpt_oss_vllm.py --config configs/metadata_config_gpt_oss_120b.json 

# Sanity check the endpoint
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","prompt":"The future of AI is","max_tokens":100,"temperature":0}'
```

## Dataset Preparation
Download validation/test JSONL from Kaggle: https://www.kaggle.com/competitions/cure-bench

Example config snippets:
```json
{
  "dataset": {
    "dataset_name": "cure_bench_phase_1",
    "dataset_path": "/abs/path/curebench_valset_phase1.jsonl",
    "description": "CURE-Bench 2025 val questions"
  }
}
```
Replicate for `curebench_testset_phase2.jsonl` when generating final submissions. All configs live in `configs/metadata_config_*.json`. Use absolute paths or paths relative to repo root.

## Supported Models

### Open-source (agentic + internal reasoning)
| Model | Type | Tools | Config |
| :---: | :---: | :---: | :---: |
| TxAgent (Llama-3.1-8B) | Agentic | ✅ | `metadata_config_testset_phase2_txagent.json` |
| Llama-3.1-8B / 70B | Internal | ❌ | `metadata_config_testset_phase2_llama3_8b.json` |
| Llama3-Med42-8B | Internal | ❌ | `metadata_config_testset_phase2_llama3_med42_8b.json` |
| Qwen3-8B | Internal | ❌ | `metadata_config_testset_phase2_qwen3_8b.json` |
| Qwen3-32B-Medical | Internal | ❌ | `mmetadata_config_testset_phase2_qwen3_32b_medical.json` |
| Baichuan-M2-32B | Internal | ❌ | `metadata_config_testset_phase2_baichuan_m2_32b.json` |
| MedGemma-4B / 27B | Internal | ❌ | `metadata_config_testset_phase2_medgemma_4b_it.json` |
| GPT-OSS 20B / 120B | Internal | ❌ | `metadata_config_testset_phase2_gpt_oss_20b.json` |

### Closed-source (Gemini)
| Model | Type | Search | 
| :---: | :---: | :---: | 
| gemini-2.5-flash | API | ✅ |
| gemini-2.5-pro | API | ✅ | 
| gemini-3-pro-preview* | API | ✅ |  

\*Post-competition benchmark used to validate scalability (see Tech report Table 2).

## Inference

### Single-agent pipelines

**TxAgent + other open-source configs**
```bash
export CONFIG_DIR=configs
python run.py --config "${CONFIG_DIR}/metadata_config_testset_phase2_txagent.json.json"
python run.py --config "${CONFIG_DIR}/metadata_config_testset_phase2_llama3_8b.json"
# ...repeat for any config listed in configs/
```
All runs write `submission.csv` + metadata JSON under `results/competition_testset_<phasex>_<model_name>/`.

**GPT-OSS via vLLM**
```bash
# Curl sanity check (replace port/model as needed)
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-20b","prompt":"The future of AI is","max_tokens":100,"temperature":0}'

# Batch evaluation (vLLM serving)
python run_gpt_oss_vllm.py --config configs/metadata_config_gpt_oss_20b.json --verbose
```

**Gemini-only API**  
1. Export `GEMINI_API_KEY` (and set `modelmodel_name` `google_search_enabled` `num_workers` etc in `gemini_with_search_config.json` if needed).  
2. Example config:
```json
{
  "dataset_path": "dataset/curebench_testset_phase2.jsonl",
  "num_workers": 8,
  "full_evaluation": true,
  "google_search": true,
  "model_name": "gemini-2.5-flash",
  "output_dir": "gemini/testset_gemini_2.5_flash_with_search_phase2_results"
}
```
3. Launch inference:
```bash
cd gemini
python run_testset_with_search.py
```
The script spawns per-worker Gemini clients, streams progress, and writes `testset_submission_<timestamp>.csv` + zipped metadata inside the configured `output_dir` in `gemini_with_search_config.json`.

### Multi-agent Executor-Analyst pipeline (TxAgent ✕ Gemini)
Our winning submission uses a stratified late-fusion ensemble:

1. **Executor self-consistency** – Run TxAgent with temperature `T=0.8` and sample budget `n=10` (or `n=10×3` for the ensemble) to harvest diverse tool-use transcripts. See Tech report Tables 1–2 for scaling gains.
2. **Evidence aggregation** – Each TxAgent subgroup writes its own CSV in `results/`. Retain the top-k tool calls per subgroup rather than pooling globally to preserve rare but critical evidence.
3. **Analyst reasoning & post-processing (Gemini)** – Use `gemini/run_final_step_with_gemini.py` to rerank each subgroup’s `submission.csv`:
   ```bash
   cd gemini
   python run_final_step_with_gemini.py \
     /root/code/CureAgent/results/<txagent_run>/submission.csv \
     --model_name gemini-2.5-flash \
     --enable_search \
     --api_key ${GEMINI_API_KEY} \
     --output stratified_group1.csv
   ```
   The script removes invalid `tool` messages, truncates long traces, retries with Gemini, extracts `[FinalAnswer]` tags (or falls back to A/B/C/D heuristics), and supports multiprocessing via `--num_workers`.
4. **Late fusion** – Majority vote across Analyst outputs (choices + rationales). Combine `stratified_group*.csv`, and package the final Kaggle ZIP.

This topology prevents the early information bottleneck noted in Tech Report (§Methods, Fig. 3) and underpins our Track 2 results.

## Results Snapshot
- **Open-source**: TxAgent self-consistency (n=30) reaches 73.5% on phase2; other OSS models lag by 15–40 pts without fine-tuned tool use (see Tech report, Table 1).
- **Closed-source**: Gemini-2.5-Pro + search hits 74.8% standalone; Gemini-3-Pro with search climbs to 81.3%.
- **Executor-Analyst (ours)**: TxAgent (10×3) + Gemini-2.5 Flash (search)x3 + stratified late fusion delivers **83.8%** phase2 accuracy, securing 2nd place Track 2.

## Acknowledgement

- AMD for providing GPU resources, and https://www.synapnote.ai/
- [CUREBench](https://github.com/mims-harvard/CUREBench) — parts of the code are derived from their baseline

## Contact Info
- helloxietingting@gmail.com

## Citation
Please cite the upcoming tech report once the arXiv link is live:
```
@article{xie2025cureagent,
  title={CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning},
  author={Ting-Ting Xie and Yixin Zhang},
  journal={arXiv preprint (Submitted, waiting for id)},
  year={2025}
}
```
