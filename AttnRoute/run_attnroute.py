import torch
import shutil
import argparse
from time import time
from typing import Tuple, Union
import pandas as pd
import numpy as np
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template

from attnroute.hparams import AttnRouteHyperParams
from attnroute.attnroute_main import (
    AttnRouteProgram,
    apply_attnroute_to_model,
    build_attnroute_subspace,
)
from dsets import MultiHarmbenchDataset
from util.globals import *

ALG_DICT = {
    "ATTNROUTE": (AttnRouteHyperParams, apply_attnroute_to_model),
}
DS_DICT = {
    "HarmBench": (MultiHarmbenchDataset),
}


def print_gpu_mem(tag):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f">>> [GPU MEM] {tag} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


def parse_layers(s: str):
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def set_global_seed(seed: int):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    data_name: str,
    dataset_size_limit: int,
    conserve_memory: bool,
    model_path: str = None,
    num_batch: int = 0,
    save_model: bool = True,
    out_name: str = None,
    edit_mode: str = "attnroute",
    fixed_target_layers: str = "",
    norm_ref_file: str = "",
    edit_data_file: str = "data/merged_output_mistral.csv",
    eta: float = 0.5,
    alpha_max: float = 2.5,
    cov_mode: str = "full",
    cov_damping: float = 1e-2,
    seed: int = 42,
    rank_mode: str = "sve",
    fixed_rank: int = 1,
):
    if out_name is None:
        raise ValueError("--out_name is required")

    run_dir = RESULTS_DIR / out_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    params_path = HPARAMS_DIR / hparams_fname
    hparams = AttnRouteHyperParams.from_json(params_path)
    assert cov_mode in ("full", "none", "damped"), f"Unknown cov_mode: {cov_mode}"
    hparams.cov_mode = cov_mode
    hparams.cov_damping = cov_damping
    hparams.seed = seed
    print(f"[Ablation] cov_mode={hparams.cov_mode}, cov_damping={hparams.cov_damping}, seed={hparams.seed}")
    fixed_target_layers = parse_layers(fixed_target_layers)
    print(fixed_target_layers)
    norm_ref = None
    norm_ref_meta = None
    if norm_ref_file:
        with open(norm_ref_file, "r", encoding="utf-8") as f:
            norm_payload = json.load(f)
        if "update_norms" in norm_payload:
            norm_ref = norm_payload["update_norms"]
            norm_ref_meta = norm_payload.get("meta", {})
        else:
            norm_ref = norm_payload
            norm_ref_meta = {}

    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing AttnRoute with parameters {hparams}")

    if model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            max_memory={0: "20GiB", 1: "20GiB"}
        )
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token = tok.eos_token
        tok.add_bos_token = False
        tok.padding_side = "right"
    elif isinstance(model_name, str):
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        tok.add_bos_token = False
        tok.padding_side = "right"
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    print_gpu_mem("After Model Loading")
    set_global_seed(hparams.seed) 
    start = time()
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    if edit_mode == "random_direct":
        if not fixed_target_layers:
            raise ValueError("random_direct requires --fixed_target_layers, e.g. '12,17,23'")
        if norm_ref is None:
            raise ValueError("random_direct requires --norm_ref_file")

        if norm_ref_meta:
            ref_module = norm_ref_meta.get("rewrite_module_tmp")
            ref_layers = norm_ref_meta.get("target_layers")
            ref_mode = norm_ref_meta.get("edit_mode")
            ref_model_name = norm_ref_meta.get("model_name")

            if ref_mode is not None and ref_mode not in {"attnroute", "deco"}:
                raise ValueError(f"norm_ref_file must come from an AttnRoute run, got edit_mode={ref_mode}")
            if ref_module is not None and ref_module != hparams.rewrite_module_tmp:
                raise ValueError(
                    f"norm_ref_file module mismatch: ref={ref_module}, current={hparams.rewrite_module_tmp}"
                )
            if ref_layers is not None and list(ref_layers) != list(fixed_target_layers):
                raise ValueError(
                    f"norm_ref_file target_layers mismatch: ref={ref_layers}, current={fixed_target_layers}"
                )
            if ref_model_name is not None and ref_model_name != model_name:
                print(
                    f"[Warning] norm_ref_file model_name={ref_model_name}, "
                    f"but current model_name={model_name}"
                )

        print("========================Skipping Safety Subspace Assets for random_direct========================")
        selected_s1_layers = fixed_target_layers
        stage1_config = {
            "layers": selected_s1_layers,
            "module": hparams.rewrite_module_tmp,
            "alpha": hparams.alpha,
            "beta": hparams.beta,
            "mom2": hparams.mom2_update_weight,
        }
        attn_assets = None

    else:
        print("========================Building or Loading Safety Subspace Assets...========================")
        probe_layers = list(range(10, 21))
        print("\n" + "=" * 30 + " PART A: ATTENTION ASSETS " + "=" * 30)
        attn_assets_path = cache_dir / "attn_safety_assets.pt"

        if attn_assets_path.exists():
            print(f"Loading cached ATTENTION assets from {attn_assets_path}")
            attn_assets = torch.load(attn_assets_path)
        else:
            print(f"Building ATTENTION assets ({hparams.rewrite_module_tmp})...")
            attn_assets = build_attnroute_subspace(
                model,
                model_name,
                tok,
                hparams,
                device="cuda",
                all_target_layers=probe_layers,
                target_module_fmt=hparams.rewrite_module_tmp,
                data_type="harmful",
                rank_mode = rank_mode,
                fixed_rank = fixed_rank
            )
            torch.save(attn_assets, attn_assets_path)

        print_gpu_mem("After Subspace Assets Build/Load")
        print(">>>> Auto-Selecting Stage 1 Layers (Based on SNR Gradient)...")
        attn_indices = sorted(attn_assets.keys())
        attn_snrs = [attn_assets[l]["snr_score"] for l in attn_indices]
        s1_candidates_idx = np.argsort(attn_snrs)[-5:]
        selected_s1_layers = sorted([attn_indices[i] for i in s1_candidates_idx])
        print(f"  [Result] Selected Stage 1 Layers: {selected_s1_layers}")

        stage1_config = {
            "layers": selected_s1_layers,
            "module": hparams.rewrite_module_tmp,
            "alpha": hparams.alpha,
            "beta": hparams.beta,
            "mom2": hparams.mom2_update_weight,
        }

    print("\nLoading ORIGINAL HARMFUL data for editing...")
    df = pd.read_csv(edit_data_file)
    edit_data = df["prompt"].tolist()

    def format_prompt(p):
        return tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )

    edit_data = [format_prompt(p) for p in edit_data]

    print("\n\n################################################")
    print("        STARTING STAGE 1: ATTENTION EDIT")
    print("################################################")
    print_gpu_mem("Before apply_attnroute")

    if edit_mode == "random_direct":
        safety_programs = None
    else:
        safety_programs = {
            layer: AttnRouteProgram(attn_assets[layer], hparams, device="cuda")
            for layer in selected_s1_layers
        }

    edited_model = model
    runtime_edit_mode = "attnroute" if edit_mode == "reroute" else edit_mode
    edited_model, weights_copy, update_norms = apply_attnroute_to_model(
        edited_model,
        tok,
        requests=edit_data,
        hparams=hparams,
        safety_programs=safety_programs,
        rewrite_module_tmp=stage1_config["module"],
        layer_module_tmp=stage1_config["module"],
        mom2_update_weight=stage1_config["mom2"],
        target_layers=stage1_config["layers"],
        copy=False,
        return_orig_weights=True,
        edit_mode=runtime_edit_mode,
        norm_ref=norm_ref,
        stats_out_path=str(run_dir / "cov_ablation_stats.json"), 
    )

    norm_payload = {
        "meta": {
            "model_name": model_name,
            "hparams_fname": hparams_fname,
            "rewrite_module_tmp": stage1_config["module"],
            "target_layers": stage1_config["layers"],
            "edit_mode": runtime_edit_mode,
        },
        "update_norms": update_norms,
    }
    with open(run_dir / "update_norms.json", "w", encoding="utf-8") as f:
        json.dump(norm_payload, f, indent=2)

    all_exec_time = time() - start
    torch.cuda.empty_cache()
    print_gpu_mem("After Stage 1 and Cache Empty")

    if save_model:
        save_dir = run_dir / "edited_model"
        print(f"Saving final edited model to {save_dir}")

        if hasattr(edited_model, "generation_config") and edited_model.generation_config is not None:
            gen_cfg = edited_model.generation_config
            if getattr(gen_cfg, "do_sample", False) is False:
                if hasattr(gen_cfg, "temperature"):
                    gen_cfg.temperature = None
                if hasattr(gen_cfg, "top_p"):
                    gen_cfg.top_p = None
                if hasattr(gen_cfg, "top_k"):
                    gen_cfg.top_k = None

        edited_model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print("Model saved successfully.")

    print("***************************************")
    print("all time is", all_exec_time)
    print("***************************************")
    print("Model editing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Model to apply the algorithm to.")
    parser.add_argument("--hparams_fname", type=str, required=True, help="Name of hyperparameters file.")
    parser.add_argument("--ds_name", default="HarmBench", choices=["HarmBench"], help="Dataset to perform edits on.")
    parser.add_argument("--data_name", type=str, default="HarmBench.json", help="Specific dataset name to load.")
    parser.add_argument("--dataset_size_limit", type=int, default=200, help="Truncate dataset to first n records.")
    parser.add_argument("--conserve_memory", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_path", type=str, default=None, help="Path of model/tokenizer.")
    parser.add_argument("--out_name", type=str, default=None, help="The out dir name for results.")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--edit_mode", type=str, default="attnroute", choices=["attnroute", "deco", "random_direct"])
    parser.add_argument("--fixed_target_layers", type=str, default="")
    parser.add_argument("--norm_ref_file", type=str, default="")
    parser.add_argument("--edit_data_file", type=str, default="data/merged_output_vicuna.csv")
    parser.add_argument("--eta", type=float, default=0.5, help="SVE information retention threshold.")
    parser.add_argument("--alpha_max", type=float, default=2.5, help="Dynamic gain coefficient.")
    parser.add_argument("--cov_mode", type=str, default="full",
                        choices=["full", "none", "damped"],
                        help="Covariance regularization ablation arm.")
    parser.add_argument("--cov_damping", type=float, default=1e-2,
                        help="Damping coefficient for cov_mode=damped.")
    parser.add_argument("--seed", type=int, default=44,
                        help="Global seed for reproducible ablation.")
    parser.add_argument("--rank_mode", type=str, default="sve", choices=["sve", "fixed"])
    parser.add_argument("--fixed_rank", type=int, default=32)
    args = parser.parse_args()

    main(
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.data_name,
        args.dataset_size_limit,
        args.conserve_memory,
        model_path=args.model_path,
        out_name=args.out_name,
        save_model=args.save_model,
        edit_mode=args.edit_mode,
        fixed_target_layers=args.fixed_target_layers,
        norm_ref_file=args.norm_ref_file,
        edit_data_file=args.edit_data_file,
        eta=args.eta,                   # 【新增】
        alpha_max=args.alpha_max,
        cov_mode=args.cov_mode,          # 新增
        cov_damping=args.cov_damping,    # 新增
        seed=args.seed,
        rank_mode=args.rank_mode,
        fixed_rank=args.fixed_rank,
    )


