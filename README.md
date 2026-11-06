# AttnRoute

AttnRoute is a post-hoc safety editing method for decoder-only large language models. It edits attention output projection weights (`W_o`) to reroute harmful semantic dependencies induced by jailbreak prompts, without adding an inference-time defense module.

This repository is organized around two complementary roles:

- **AttnRoute edits the model.** It builds safety subspaces, reconstructs safer target activations, and writes an edited Hugging Face checkpoint.
- **HarmBench tests the edited model.** Red-team prompt generation, harmfulness classification, and ASR computation should be performed with the official HarmBench evaluation pipeline.

The codebase includes the AttnRoute editing pipeline, ROME-style covariance utilities, HarmBench-related behavior data, model hyperparameters, and plotting scripts for ablations and result figures. It does **not** provide a full built-in HarmBench evaluator.

## Repository Layout

- `run_attnroute.py`: main entry point for editing a Hugging Face causal language model.
- `attnroute/`: AttnRoute subspace construction, activation extraction, target optimization, and closed-form weight-update implementation.
- `rome/`: covariance/statistics utilities used by the closed-form update.
- `dsets/`: dataset wrappers used by the editing utilities.
- `data/`: harmful, harmless, jailbreak, refusal, and HarmBench behavior/source data.
- `hparams/`: JSON configurations for Vicuna, Mistral, and Llama checkpoints.
- `cagrad/`: CAGrad implementation and saved optimization histories.
- `ablation/`: standalone scripts for alpha/eta/rank ablation figures.
- `debug_tokens.py`: tokenizer/BOS diagnostic utility.

## Workflow Overview

The recommended workflow is:

```text
1. Prepare a supported base model and editing data.
2. Run AttnRoute to edit the model.
3. Save the edited checkpoint under results/<out_name>/edited_model/.
4. Use the official HarmBench pipeline to red-team test that edited checkpoint.
5. Report ASR from HarmBench and utility metrics from the chosen utility benchmark.
```

AttnRoute itself is responsible only for step 2 and checkpoint export. HarmBench is responsible for attack generation, judging, and ASR calculation.

## Method Summary

AttnRoute edits the attention output projection module, normally:

```text
model.layers.{}.self_attn.o_proj
```

The current pipeline performs the following operations:

1. Loads harmful and harmless prompts and extracts activations from candidate transformer layers.
2. Constructs harmful-versus-harmless activation differences using nearest-neighbor matching.
3. Applies SVD to localize safety-relevant directions and scores candidate layers.
4. Selects the five highest-scoring layers from the default probe range, layers 10 through 20.
5. Reconstructs safer target activations with conflict-aware optimization.
6. Computes covariance-regularized closed-form updates to `W_o`.
7. Writes the edited model, run hyperparameters, cached subspace assets, and update norms to `results/<out_name>/`.

After editing, the model remains an ordinary Hugging Face checkpoint. No wrapper, detector, decoding modification, or runtime intervention is required at inference time.

## Requirements

The editing pipeline is intended for CUDA-capable machines. It loads models in `bfloat16` and uses `device_map="auto"`; GPU memory requirements depend on the checkpoint, prompt count, and covariance-statistics settings.

Required baseline environment:

- Python 3.12.3
- NVIDIA GPU and CUDA-enabled PyTorch
- A causal language-model checkpoint compatible with `transformers`

Create an environment and install the pinned dependencies:

```bash
conda create -n attnroute python=3.12.3 -y
conda activate attnroute
pip install -r requirements.txt
```

Several modules additionally import packages that may not be pinned in `requirements.txt`:

```bash
pip install fastchat scipy scikit-learn pandas tqdm seaborn
```

For the ROME covariance-statistics utilities, ensure the required statistics data can be downloaded or is available under `data/stats1/`. The location is configured in `util/globals.py`.

## Model Checkpoints

The entry script accepts either a Hugging Face model identifier or a local checkpoint directory.

Supported hyperparameter templates are currently provided for:

- `vicuna-7b-v1.5`
- `Mistral-7B-Instruct-v0.2`
- `Llama-3.1-8B-Instruct`

Before editing a different architecture, verify the attention output-projection path. The default `model.layers.{}.self_attn.o_proj` template is appropriate for many Llama-family and Mistral-family checkpoints, but it is not universal.

## Data

### Editing prompts

The main script reads a CSV file with a required `prompt` column. Its default is:

```text
data/merged_output_mistral.csv
```

Other included prompt files are:

| File                             | Intended content                 |
| -------------------------------- | -------------------------------- |
| `data/merged_output_vicuna.csv`  | Default vicuna editing prompts   |
| `data/merged_output_mistral.csv` | Mistral-oriented editing prompts |
| `data/merged_output_llama.csv`   | Llama-oriented editing prompts   |
| `data/harmful.csv`               | Harmful prompts                  |
| `data/harmless.csv`              | Harmless prompts                 |
| `data/jailbreak_prompts.csv`     | Jailbreak prompts                |
| `data/refusal.csv`               | Refusal examples                 |

### Subspace construction data

During standard AttnRoute editing, `attnroute/attnroute_main.py` reads:

- `data/harmbench_behaviors_text_all.csv`, which must provide `Behavior` and `target` columns;
- `data/harmless.csv`, which must provide a `prompt` column.

The harmful behaviors are formatted with the model chat template. The implementation also combines them with refusal and compliance targets when constructing the safety subspace.

## Running AttnRoute Editing

Run commands from the repository root. The hyperparameter files under `hparams/` use the naming pattern `<model>-<locus>.json`, where the plain `<model>.json` targets `W_o`, and the `-wv` / `-mlp` / `-rand` variants are the locus-ablation configurations.

### Minimal local command

```bash
python run_attnroute.py \
  --model_name mistral \
  --hparams_fname Mistral-7B-Instruct-v0.2.json \
  --out_name mistral_attnroute \
  --model_path "../models/mistral/Mistral-7B-Instruct-v0.2" \
  --edit_mode attnroute \
  --edit_data_file data/merged_output_mistral.csv
```

`--model_path` points to the local checkpoint and accepts relative paths (as used by the internal runs). If `--model_path` is omitted, `--model_name` must be a Hugging Face identifier and the checkpoint is downloaded instead:

```bash
python run_attnroute.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --hparams_fname Mistral-7B-Instruct-v0.2.json \
  --out_name mistral_attnroute \
  --edit_mode attnroute
```

### One command per supported model

```bash
# Mistral-7B-Instruct-v0.2
python run_attnroute.py --model_name mistral \
  --hparams_fname Mistral-7B-Instruct-v0.2.json \
  --out_name mistral_attnroute --edit_mode attnroute \
  --model_path "../models/mistral/Mistral-7B-Instruct-v0.2" \
  --edit_data_file data/merged_output_mistral.csv

# Vicuna-7B-v1.5
python run_attnroute.py --model_name vicuna \
  --hparams_fname vicuna-7b-v1.5.json \
  --out_name vicuna_attnroute --edit_mode attnroute \
  --model_path "../models/vicuna/vicuna-7b-v1.5" \
  --edit_data_file data/merged_output_vicuna.csv

# Llama-3.1-8B-Instruct
python run_attnroute.py --model_name llama \
  --hparams_fname Llama-3.1-8B-Instruct.json \
  --out_name llama_attnroute --edit_mode attnroute \
  --model_path "../models/llama/Llama-3.1-8B-Instruct" \
  --edit_data_file data/merged_output_llama.csv
```

### Cluster submission

`run_attnroute.sbatch` shows a full Slurm job: it activates the conda environment, sets CUDA paths, and runs the edit with explicit ablation flags. Submit it with:

```bash
sbatch run_attnroute.sbatch
```

The sbatch example also demonstrates the rank ablation arm (`--rank_mode fixed --fixed_rank 128`) and how `--fixed_target_layers` / `--norm_ref_file` pin the layer choice and update scale to a reference run:

```bash
python run_attnroute.py \
  --model_name mistral \
  --hparams_fname Mistral-7B-Instruct-v0.2.json \
  --out_name module_ablation/static-rank128/Mistral-7B-Instruct-v0.2 \
  --model_path "../models/mistral/Mistral-7B-Instruct-v0.2" \
  --edit_data_file "data/merged_output_mistral.csv" \
  --fixed_target_layers "12,13,14,16,20" \
  --norm_ref_file "./results/module_ablation/Mistral-7B-Instruct-v0.2/update_norms.json" \
  --edit_mode attnroute \
  --cov_mode full \
  --rank_mode "fixed" \
  --fixed_rank 128
```

### Locus-ablation edits (W_v / MLP)

Same entry point, different hyperparameter file:

```bash
python run_attnroute.py --model_name vicuna \
  --hparams_fname vicuna-7b-v1.5-wv.json  --out_name vicuna_wv   --edit_mode attnroute \
  --model_path "../models/vicuna/vicuna-7b-v1.5"
python run_attnroute.py --model_name vicuna \
  --hparams_fname vicuna-7b-v1.5-mlp.json --out_name vicuna_mlp  --edit_mode attnroute \
  --model_path "../models/vicuna/vicuna-7b-v1.5"
```

### Command-line arguments

| Argument                | Required            | Description                                                 |
| ----------------------- | ------------------- | ----------------------------------------------------------- |
| `--model_name`          | Yes                 | Hugging Face identifier or model label                      |
| `--hparams_fname`       | Yes                 | Hyperparameter JSON file under `hparams/`                   |
| `--out_name`            | Yes                 | Name of the run directory under `results/`                  |
| `--model_path`          | No                  | Local model/tokenizer directory                             |
| `--edit_mode`           | No                  | `attnroute`,  or `random_direct`; default: `attnroute`      |
| `--edit_data_file`      | No                  | Editing CSV with a `prompt` column                          |
| `--eta`                 | No                  | SVE information-retention threshold; default: `0.5`         |
| `--alpha_max`           | No                  | Dynamic gain coefficient; default: `2.5`                    |
| `--cov_mode`            | No                  | `full`, `none`, or `damped`; default: `full`                |
| `--cov_damping`         | No                  | Scalar damping strength when `cov_mode=damped`              |
| `--rank_mode`           | No                  | `sve` (read rank from spectrum) or `fixed`; default: `sve`  |
| `--fixed_rank`          | No                  | Subspace rank when `rank_mode=fixed`; default: `32`         |
| `--seed`                | No                  | Global random seed; default: `42`                           |
| `--save_model`          | No                  | Save the edited model; enabled by default                   |
| `--no-save_model`       | No                  | Do not save the edited model                                |
| `--fixed_target_layers` | For `random_direct` | Comma-separated layer indices, for example `12,15,16,17,18` |
| `--norm_ref_file`       | For `random_direct` | `update_norms.json` from a compatible AttnRoute run         |


The legacy name `reroute` appears in older code paths but is not accepted by the current command-line parser. Use `--edit_mode attnroute`.

## Output Layout

A run using `--out_name vicuna_attnroute` creates:

```text
results/
  vicuna_attnroute/
    params.json
    update_norms.json
    cache/
      attn_safety_assets.pt
    edited_model/
```

- `params.json`: copy of the hyperparameter JSON used by the run.
- `update_norms.json`: update magnitudes plus metadata about the edited layers and module.
- `cache/attn_safety_assets.pt`: cached activation/subspace assets. A later run with the same output name reuses this cache.
- `edited_model/`: edited model and tokenizer in Hugging Face `save_pretrained` format.

Delete `cache/attn_safety_assets.pt` if the model, data, candidate layers, or subspace-related hyperparameters change.

## HarmBench Red-Team Evaluation

AttnRoute does not replace HarmBench. After editing, evaluate the saved checkpoint with the HarmBench codebase (a local clone is expected as a sibling directory, e.g. `../HarmBench`).

Use this edited checkpoint as the model under test:

```text
results/<out_name>/edited_model/
```

### Step 0: Register the edited model in HarmBench

Add an entry to `../HarmBench/configs/model_configs/models.yaml` whose `model_name_or_path` points to the edited checkpoint:

```yaml
mistral_7b_attnroute:
  model:
    model_name_or_path: "/abs/path/to/AttnRoute/results/mistral_attnroute/edited_model"
    use_fast_tokenizer: False
  model_type: "open_source"
  num_gpus: 1
```

All `generate_completions.py` commands below reference this registered name via `--model_name`.

### Step 1: Generate attack test cases

Run from the HarmBench repository root. This produces the per-attack test cases under the given `--save_dir`; for adaptive attacks, re-run the attack optimization directly against the registered edited model:

```bash
cd ../HarmBench
python generate_test_cases.py \
  --method_name AutoDAN \
  --experiment_name mistral_7b_attnroute \
  --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --save_dir ./results/mistral_autodan_harmbench
```

Repeat per attack family (`GCG`, `AutoDAN`, `PAIR`). The corresponding Slurm jobs are in `../HarmBench/generate_cases.sbatch`:

```bash
sbatch generate_cases.sbatch
```

### Step 2: Generate completions with the edited model

Feed each attack's test cases to the edited model and save completions:

```bash
python generate_completions.py \
  --model_name mistral_7b_attnroute \
  --models_config_file ./configs/model_configs/models.yaml \
  --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --test_cases_path ./results/mistral_autodan_harmbench/test_cases/test_cases.json \
  --save_path ./results/mistral-harmbench_autodan_attnroute.json \
  --max_new_tokens 512 \
  --generate_with_vllm \
  --incremental_update
```

- `--generate_with_vllm`: use vLLM for batched decoding.
- `--incremental_update`: resume into the same `--save_path` instead of regenerating from scratch.
- Name the output JSON as `<model>-<benchmark>_<attack>_<variant>.json` (e.g. `llama-harmbench_gcg_wo.json`); the ASR script in Step 3 parses this pattern.

The batch jobs for all model × benchmark × attack combinations are in `../HarmBench/generate_completions.sbatch`:

```bash
sbatch generate_completions.sbatch
```

### Step 3: Classify completions and compute ASR

Run the batch classifier over the saved completion JSONs:

```bash
python batch_asr.py
# or on the cluster:
sbatch eval_completions.sbatch
```

`batch_asr.py` scans a results directory recursively, matches filenames against `<model>-<benchmark>_<attack>_<variant>.json`, judges each `(behavior, generation)` pair with the local HarmBench classifier (`HarmBench-Llama-2-13b-cls`), and appends per-file ASR to a results CSV. It skips files already present in the CSV, so reruns are incremental. Edit `JSON_ROOT` and `RESULT_CSV` at the top of the script to point at your completion directory and output file.

The intended safety metric is HarmBench attack success rate (ASR):

```text
ASR = number of classifier-positive harmful generations / total evaluated generations
```

Lower ASR indicates stronger jailbreak resistance. For a fair comparison, keep the HarmBench behavior set, jailbreak templates, generation parameters, classifier checkpoint, and decoding method fixed across all methods.

This repository includes HarmBench-related behavior and target data under `data/`, but the evaluator itself is intentionally not part of this codebase.

## Random Direct Baseline

`random_direct` bypasses AttnRoute subspace construction and applies direct updates to manually specified layers. It requires a compatible `update_norms.json` file from an AttnRoute run:

```bash
python run_attnroute.py \
  --model_name vicuna \
  --model_path "../models/vicuna/vicuna-7b-v1.5" \
  --hparams_fname vicuna-7b-v1.5.json \
  --out_name vicuna_random_direct \
  --edit_mode random_direct \
  --fixed_target_layers 12,15,16,17,18 \
  --norm_ref_file results/vicuna_attnroute/update_norms.json
```

The script validates the reference model name, edited module template, selected layers, and edit mode when this metadata is available.

## Hyperparameters

The currently used `AttnRouteHyperParams` fields are:

| Field                | Description                                       |
| -------------------- | ------------------------------------------------- |
| `fact_token`         | Token position used for activation extraction     |
| `mom2_update_weight` | Covariance-regularization strength                |
| `rewrite_module_tmp` | Weight module template to edit                    |
| `layer_module_tmp`   | Module template used by layer utilities           |
| `mom2_dataset`       | Dataset used to estimate second-moment statistics |
| `mom2_n_samples`     | Number of statistics samples                      |
| `mom2_dtype`         | Statistics precision                              |
| `alpha`              | Target reconstruction weight                      |
| `beta`               | Safety-subspace weight                            |
| `maxdist`            | Optimization distance constraint                  |
| `lr`                 | Optimization learning rate                        |

Use a matching configuration file for the checkpoint architecture. Some JSON files in `hparams/` contain fields from older experiments; `run_attnroute.py` uses the fields defined in `attnroute/hparams.py`.

## Troubleshooting

### CUDA out of memory

- Use a larger GPU or multiple visible GPUs.

### Model module cannot be found

Check `rewrite_module_tmp` in the selected hyperparameter JSON. It must exactly match a named module in the checkpoint, including the layer prefix and projection name.

### Missing data columns

- The editing CSV requires `prompt`.
- `data/harmbench_behaviors_text_all.csv` requires `Behavior` and `target`.
- `data/harmless.csv` requires `prompt`.

### Chat-template errors

The pipeline calls `tokenizer.apply_chat_template`. Use a tokenizer with a valid chat template, or adapt prompt formatting in `run_attnroute.py` and `attnroute/attnroute_main.py` for the target checkpoint.

## Citation and License

Add the final paper citation and project license before public release. Also verify the redistribution terms for model checkpoints and bundled datasets.
