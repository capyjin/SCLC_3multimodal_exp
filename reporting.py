# -*- coding: utf-8 -*-
"""Writes the output-folder layout mandated by
``DATA/SCLC_EXPERIMENT_PROTOCOL.md`` section 11:

outputs/EXP_YYYYMMDD_NNN_name/
    resolved_config.yaml, data_manifest.csv, splits.csv, environment.txt,
    checkpoints/, logs/, metrics/{fold_metrics.csv,oof_predictions.csv,summary.json},
    plots/, experiment_report.md
"""
import datetime as _dt
import json
import os
import platform
import sys

import pandas as pd
import yaml

import cohort


def prepare_experiment_dir(output_root: str, experiment_id: str) -> dict[str, str]:
    root = os.path.join(output_root, experiment_id)
    dirs = {
        "root": root,
        "checkpoints": os.path.join(root, "checkpoints"),
        "logs": os.path.join(root, "logs"),
        "metrics": os.path.join(root, "metrics"),
        "plots": os.path.join(root, "plots"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def write_resolved_config(dirs: dict[str, str], contract: dict) -> str:
    path = os.path.join(dirs["root"], "resolved_config.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(contract, fh, allow_unicode=True, sort_keys=False)
    return path


def write_data_manifest(dirs: dict[str, str]) -> str:
    manifest = cohort.build_manifest()
    path = os.path.join(dirs["root"], "data_manifest.csv")
    manifest.to_csv(path, index=False)
    return path


def write_splits(dirs: dict[str, str], split_csv: str = cohort.DEFAULT_SPLIT_CSV) -> str:
    split = pd.read_csv(split_csv)
    path = os.path.join(dirs["root"], "splits.csv")
    split.to_csv(path, index=False)
    return path


def write_environment(dirs: dict[str, str]) -> str:
    lines = [
        f"python_version: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"run_date_utc: {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}",
    ]
    try:
        import torch
        lines.append(f"torch_version: {torch.__version__}")
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu_name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        lines.append("torch_version: not_installed")
    path = os.path.join(dirs["root"], "environment.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_metrics(dirs: dict[str, str], fold_records: list[dict], oof_predictions: list[dict],
                   summary: dict) -> dict[str, str]:
    fold_path = os.path.join(dirs["metrics"], "fold_metrics.csv")
    oof_path = os.path.join(dirs["metrics"], "oof_predictions.csv")
    summary_path = os.path.join(dirs["metrics"], "summary.json")
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)
    pd.DataFrame(oof_predictions).to_csv(oof_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    return {"fold_metrics": fold_path, "oof_predictions": oof_path, "summary": summary_path}


def write_experiment_report(dirs: dict[str, str], contract: dict, summaries: dict) -> str:
    lines = [
        f"# {contract.get('experiment_name', contract.get('experiment_id'))}",
        "",
        "## 1. Research question", "", contract.get("hypothesis", ""), "",
        "## 2. Compared models", "", str(contract.get("primary_comparison", "")), "",
        "## 3. Intentional change", "", str(contract.get("intentional_change", "")), "",
        "## 4. Fixed conditions", "", str(contract.get("fixed_conditions", "")), "",
        "## 5. Cohort", "",
        f"- cohort_rule: {contract.get('cohort_rule')}",
        f"- actual_n: {contract.get('actual_n')}",
        f"- split_file: {contract.get('split_file')}",
        "",
        "## 6. Results (C-index, mean +/- std)", "",
        "| target | modality | mean | std | n_folds |", "|---|---|---:|---:|---:|",
    ]
    for target, target_summaries in summaries.items():
        for modality, s in target_summaries.items():
            lines.append(f"| {target} | {modality} | {s.get('mean_c_index'):.4f} | {s.get('std_c_index'):.4f} | {s.get('n_folds')} |")
    lines += ["", "## 7. Notes", "", "See resolved_config.yaml / metrics/ for full detail.", ""]
    path = os.path.join(dirs["root"], "experiment_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def summarize(fold_records: list[dict]) -> dict:
    cis = [r["c_index"] for r in fold_records]
    if not cis:
        return {"mean_c_index": None, "std_c_index": None, "n_folds": 0}
    import numpy as np
    return {
        "mean_c_index": float(np.mean(cis)), "std_c_index": float(np.std(cis)),
        "n_folds": len(cis), "n_total": int(sum(r.get("n", 0) for r in fold_records)),
    }
