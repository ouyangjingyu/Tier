#!/usr/bin/env python3

import argparse
import gc
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import shutil


def _maybe_clear_torch_cuda_cache():
    try:
        import torch  # type: ignore
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        return


def _alpha_tag(alpha_value: float) -> str:
    return str(alpha_value).replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSFL/SplitFed/DTFL with matched settings")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "svhn", "fashion_mnist", "cinic10"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet56",
        help="Model type (e.g., resnet56, resnet110)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size",
    )
    return parser.parse_args()


def _run_one(
    *,
    label: str,
    cmd: list[str],
    cwd: str,
    output_dir: Path,
    internal_log_candidates: list[Path],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}.txt"

    start_ts = datetime.now()
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8", buffering=1) as f:
        f.write("=" * 80 + "\n")
        f.write(f"label: {label}\n")
        f.write(f"start_time: {start_ts.isoformat(timespec='seconds')}\n")
        f.write(f"cwd: {cwd}\n")
        f.write("cmd: " + " ".join(cmd) + "\n")
        f.write("=" * 80 + "\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        def _terminate_child():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)

            return_code = proc.wait()
        except KeyboardInterrupt:
            _terminate_child()
            return_code = 130
        finally:
            try:
                proc.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass

    end_ts = datetime.now()
    duration_s = time.time() - start_time

    with open(output_path, "a", encoding="utf-8", buffering=1) as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"end_time: {end_ts.isoformat(timespec='seconds')}\n")
        f.write(f"duration_seconds: {duration_s:.2f}\n")
        f.write(f"return_code: {return_code}\n")
        f.write("=" * 80 + "\n")

    for p in internal_log_candidates:
        try:
            if p.exists():
                dest = output_dir / p.name
                if dest.exists():
                    stem = dest.stem
                    suffix = dest.suffix
                    dest = output_dir / f"{stem}_{int(time.time())}{suffix}"
                shutil.copy2(p, dest)
        except Exception:
            pass

    gc.collect()
    _maybe_clear_torch_cuda_cache()
    gc.collect()

    return return_code


def main():
    args = _parse_args()
    root_dir = Path("/home/gjm/1023")
    cwd = str(root_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root_dir / "experiment_outputs" / f"{args.dataset}_psfl_splitfed_dtfl_{timestamp}"

    dataset = str(args.dataset)
    heterogeneity_settings: list[dict[str, object]] = [
        {"name": "alpha0p3", "partition_method": "hetero", "alpha": 0.3},
        {"name": "iid_homo", "partition_method": "homo", "alpha": 100.0},
    ]

    common_args = {
        "dataset": dataset,
        "data_dir": "./data",
        "client_number": 10,
        "clients_per_round": 0,
        "participation_rate": 0.0,
        "batch_size": int(args.batch_size),
        "rounds": 2,
        "model": str(args.model),
        "seed": 42,
        "lr": 0.01,
        "warmup_epochs": 1,
        "warmup_lr": 0.01,
        "pretrain_epochs": 10,
        "pretrain_lr": 0.01,
    }

    client_number = int(common_args["client_number"])
    clients_per_round = int(common_args.get("clients_per_round", 0) or 0)
    participation_rate = float(common_args.get("participation_rate", 0.0) or 0.0)
    if clients_per_round <= 0 and participation_rate > 0:
        clients_per_round = int(round(client_number * participation_rate))
        clients_per_round = max(1, min(client_number, clients_per_round))

    algorithms: list[dict] = [
        {
            "name": "psfl",
            "script": str(root_dir / "main.py"),
            "extra_args": [
                f"--running_name=PSFL_{dataset}",
                f"--dataset={common_args['dataset']}",
                f"--data_dir={common_args['data_dir']}",
                f"--client_number={common_args['client_number']}",
                f"--clients_per_round={clients_per_round}",
                f"--batch_size={common_args['batch_size']}",
                f"--rounds={common_args['rounds']}",
                f"--model={common_args['model']}",
                f"--seed={common_args['seed']}",
                f"--lr={common_args['lr']}",
                f"--warmup_epochs={common_args['warmup_epochs']}",
                f"--warmup_lr={common_args['warmup_lr']}",
            ],
            "log_candidates": lambda alpha: [
                root_dir
                / f"{dataset}_clients{common_args['client_number']}_{common_args['model']}_alpha{_alpha_tag(alpha)}.txt"
            ],
        },
        {
            "name": "splitfed",
            "script": str(root_dir / "splitfed_main_matched.py"),
            "extra_args": [
                f"--dataset={common_args['dataset']}",
                f"--data_dir={common_args['data_dir']}",
                f"--client_number={common_args['client_number']}",
                f"--clients_per_round={clients_per_round}",
                f"--batch_size={common_args['batch_size']}",
                f"--rounds={common_args['rounds']}",
                f"--model={common_args['model']}",
                f"--seed={common_args['seed']}",
                f"--lr={common_args['lr']}",
                f"--pretrain_epochs={common_args['pretrain_epochs']}",
                f"--pretrain_lr={common_args['pretrain_lr']}",
            ],
            "log_candidates": lambda alpha: [
                root_dir
                / f"{dataset}_{common_args['model']}_splitfed_matched_alpha{_alpha_tag(alpha)}.txt"
            ],
        },
        {
            "name": "dtfl",
            "script": str(root_dir / "DTFL_迪利克雷分布_测试平均值" / "main_warmup.py"),
            "extra_args": [
                f"--running_name=DTFL_{dataset}",
                f"--dataset={common_args['dataset']}",
                f"--data_dir={common_args['data_dir']}",
                f"--client_number={common_args['client_number']}",
                f"--clients_per_round={clients_per_round}",
                f"--batch_size={common_args['batch_size']}",
                f"--rounds={common_args['rounds']}",
                f"--model={common_args['model']}",
                f"--seed={common_args['seed']}",
                f"--lr={common_args['lr']}",
                f"--warmup_epochs={common_args['warmup_epochs']}",
            ],
            "log_candidates": lambda alpha: [
                root_dir
                / "DTFL_迪利克雷分布_测试平均值"
                / "logs"
                / f"DTFL_{common_args['model']}_{dataset}_alpha{alpha}.txt"
            ],
        },
    ]

    total = len(algorithms) * len(heterogeneity_settings)
    idx = 0
    failures: list[tuple[str, int]] = []

    for algo in algorithms:
        for setting in heterogeneity_settings:
            setting_name = str(setting["name"])
            partition_method = str(setting["partition_method"])
            alpha = float(setting["alpha"])
            idx += 1
            label = f"{algo['name']}_{dataset}_{setting_name}"
            print("\n" + "=" * 80)
            print(f"[{idx}/{total}] {label}")
            print("=" * 80)

            cmd = [
                sys.executable,
                algo["script"],
                f"--partition_method={partition_method}",
                f"--partition_alpha={alpha}",
                *algo["extra_args"],
            ]

            rc = _run_one(
                label=label,
                cmd=cmd,
                cwd=cwd,
                output_dir=output_dir,
                internal_log_candidates=list(algo["log_candidates"](alpha)),
            )

            if rc != 0:
                print(f"实验失败: {label} (return_code={rc})")
                failures.append((label, rc))

    print("\n全部实验完成")
    print(f"输出目录: {output_dir}")
    if failures:
        print("失败列表:")
        for name, rc in failures:
            print(f"- {name} (return_code={rc})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
