"""Portable Accelerate launcher for the published VLAFlow recipes."""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = {
    "MindPI": "train_vlaflow.py",
    "MindWPI": "train_vlaflow_wm.py",
    "MindLWPI_Compressed": "train_vlaflow_lwm.py",
}


def build_launch(args):
    config_path = Path(args.config).resolve()
    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    framework = cfg["framework"]
    model = framework["name"]
    if model not in TRAINERS:
        raise ValueError(f"Unsupported model: {model}")
    if args.mode == "frozen-vlm" and (args.phase != "pretrain" or model != "MindPI"):
        raise ValueError("frozen-vlm is only supported for MindPI pretraining")
    if args.mode == "no-pretrain" and (args.phase != "finetune" or model not in {"MindPI", "MindWPI"}):
        raise ValueError("no-pretrain is only supported for MindPI/MindWPI fine-tuning")
    cfg["run_id"] = os.environ.get("RUN_ID", cfg["run_id"] + ("_" + args.mode if args.mode != "full" else ""))
    cfg["run_root_dir"] = str(Path(os.environ.get("OUTPUT_ROOT", ROOT / "outputs")).resolve())
    cfg["datasets"]["vla_data"]["data_root_dir"] = str(Path(os.environ.get("DATA_ROOT", ROOT / "data")).resolve())
    model_root = Path(os.environ.get("MODEL_ROOT", ROOT / "models")).resolve()
    framework["qwenvl"]["base_vlm"] = str(model_root / "Qwen3-VL-4B-Instruct")
    if "latent_extractor" in framework:
        framework["latent_extractor"]["encoder_path"] = str(model_root / "vjepa2-vitl-fpc16-256")
    if framework.get("enable_lang_action"):
        framework["global_norm_stats_path"] = os.environ.get(
            "GLOBAL_NORM_STATS",
            str(Path(cfg["datasets"]["vla_data"]["data_root_dir"]) / "oxemix_global_norm_stats.json"),
        )
    cfg["trainer"]["freeze_modules"] = "qwen_vl_interface" if args.mode == "frozen-vlm" else ""
    checkpoint = os.environ.get("PRETRAINED_CHECKPOINT")
    if args.phase == "finetune" and args.mode != "no-pretrain":
        if not checkpoint and not args.dry_run:
            raise ValueError("Set PRETRAINED_CHECKPOINT to the pretrained weight file for fine-tuning")
        cfg["trainer"]["pretrained_checkpoint"] = checkpoint
    elif checkpoint:
        raise ValueError("PRETRAINED_CHECKPOINT must be unset for pretraining or no-pretrain mode")
    else:
        cfg["trainer"]["pretrained_checkpoint"] = None
    for override in args.set:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError("--set requires key=value")
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    nodes = int(os.environ.get("NUM_NODES", "1"))
    rank = int(os.environ.get("NODE_RANK", "0"))
    gpus = int(os.environ.get("GPUS_PER_NODE", "1"))
    accumulation = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", cfg["trainer"]["gradient_accumulation_steps"]))
    if nodes < 1 or gpus < 1 or accumulation < 1 or not 0 <= rank < nodes:
        raise ValueError("Invalid node, GPU, rank, or gradient accumulation setting")
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    if nodes > 1 and master in {"127.0.0.1", "localhost"}:
        raise ValueError("Multi-node training requires a reachable MASTER_ADDR")
    cfg["trainer"]["gradient_accumulation_steps"] = accumulation
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(ROOT / "vlaflow/config/zero2.yaml"),
        "--num_machines",
        str(nodes),
        "--num_processes",
        str(nodes * gpus),
        "--machine_rank",
        str(rank),
        "--main_process_ip",
        master,
        "--main_process_port",
        os.environ.get("MASTER_PORT", "29500"),
        "--gradient_accumulation_steps",
        str(accumulation),
        str(ROOT / "vlaflow/training" / TRAINERS[model]),
        "--config_yaml",
        "<effective-config.yaml>",
    ]
    batch = cfg["datasets"]["vla_data"]["per_device_batch_size"] * nodes * gpus * accumulation
    return cfg, command, batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--mode", choices=["full", "frozen-vlm", "no-pretrain"], default="full")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the effective config and command without loading models"
    )
    args = parser.parse_args()
    try:
        cfg, command, batch = build_launch(args)
        print(yaml.safe_dump(cfg, sort_keys=False), flush=True)
        print(f"Global batch: {batch}; paper reference: {cfg['paper_global_batch_size']}", flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        resources = [cfg["datasets"]["vla_data"]["data_root_dir"], cfg["framework"]["qwenvl"]["base_vlm"]]
        if "latent_extractor" in cfg["framework"]:
            resources.append(cfg["framework"]["latent_extractor"]["encoder_path"])
        for value in [cfg["trainer"].get("pretrained_checkpoint"), cfg["framework"].get("global_norm_stats_path")]:
            if value:
                resources.append(value)
        if cfg["datasets"]["vla_data"]["data_mix"] in {"oxe_mix_full", "oxe_mix_full_wm"}:
            resources.extend(
                str(Path(cfg["datasets"]["vla_data"]["data_root_dir"]) / name)
                for name in ["openx-embodiment-lerobot", "OXE_AugE_v2", "RoboCOIN_SUB"]
            )
        for resource in resources:
            if not Path(resource).exists():
                raise ValueError(
                    f"Resource does not exist: {resource}. "
                    "Configure DATA_ROOT, MODEL_ROOT, GLOBAL_NORM_STATS, or PRETRAINED_CHECKPOINT."
                )
        env = os.environ.copy()
        env["DATA_ROOT"] = cfg["datasets"]["vla_data"]["data_root_dir"]
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        with tempfile.TemporaryDirectory(prefix="vlaflow-config-") as directory:
            effective = Path(directory) / "config.yaml"
            effective.write_text(yaml.safe_dump(cfg, sort_keys=False))
            command[-1] = str(effective)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
