import argparse
import json
import os
import subprocess
from datetime import datetime, timezone


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], env: dict[str, str]) -> int:
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _detach(argv: list[str], env: dict[str, str], tag: str) -> dict:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

    ts = _ts()
    logfile = os.path.join("logs", f"optuna_full_{tag}_{ts}.log")
    pidfile = os.path.join("analysis", f"optuna_full_{tag}.pid")
    meta = os.path.join("analysis", f"optuna_full_{tag}_meta.json")

    with open(logfile, "ab", buffering=0) as lf:
        p = subprocess.Popen(
            argv,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    with open(pidfile, "w", encoding="utf-8") as f:
        f.write(str(int(p.pid)))

    info = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pid": int(p.pid),
        "logfile": logfile,
        "pidfile": pidfile,
        "cmd": argv,
    }
    _write_json(meta, info)
    return info


def main():
    ap = argparse.ArgumentParser(
        description="Run a long Optuna tuning + (optional) eval/ablation pipeline. Supports --detach."
    )
    ap.add_argument("--python", default=os.getenv("PYTHON", "python"))
    ap.add_argument("--tag", default="full")

    ap.add_argument("--optuna_trials", type=int, default=int(os.getenv("AETHER_OPTUNA_TRIALS", "200") or 200))
    ap.add_argument("--optuna_timeout_sec", type=int, default=int(os.getenv("AETHER_OPTUNA_TIMEOUT_SEC", "0") or 0))
    ap.add_argument("--optuna_storage", default=os.getenv("AETHER_OPTUNA_STORAGE", "sqlite:///analysis/optuna_full.db"))
    ap.add_argument("--optuna_study_name", default=os.getenv("AETHER_OPTUNA_STUDY_NAME", "aether_optuna_full"))

    ap.add_argument("--backtest_days", type=int, default=30)
    ap.add_argument("--backtest_stride_hours", type=int, default=4)
    ap.add_argument("--run_ablation", action="store_true")
    ap.add_argument("--ablation_days", type=int, default=7)
    ap.add_argument("--ablation_stride_hours", type=int, default=4)
    ap.add_argument("--no_telegram", action="store_true")

    ap.add_argument("--detach", action="store_true", help="Spawn in background and return immediately.")
    args = ap.parse_args()

    env = dict(os.environ)
    env["AETHER_OPTUNA_TRIALS"] = str(int(args.optuna_trials))
    env["AETHER_OPTUNA_TIMEOUT_SEC"] = str(int(args.optuna_timeout_sec))
    env["AETHER_OPTUNA_STORAGE"] = str(args.optuna_storage)
    env["AETHER_OPTUNA_STUDY_NAME"] = str(args.optuna_study_name)
    env["AETHER_OPTUNA_LOAD_IF_EXISTS"] = "1"
    env["AETHER_OPTUNA_N_JOBS"] = "1"

    pinned_end_time = datetime.now(timezone.utc).isoformat()
    env["AETHER_BACKTEST_END_TIME_ISO"] = pinned_end_time

    # We avoid network collection for robustness; trainer pulls from DB.
    train_cmd = [args.python, "main.py", "--mode", "train", "--tune", "--no_collect", "--offline_ok"]
    if args.no_telegram:
        train_cmd.append("--no_telegram")

    eval_cmd = [
        args.python,
        "scripts/eval_suite.py",
        "--days",
        str(int(args.backtest_days)),
        "--stride_hours",
        str(int(args.backtest_stride_hours)),
        "--tag",
        f"{args.tag}_eval",
    ]
    if args.no_telegram:
        eval_cmd.append("--no_telegram")

    ablation_cmd = [
        args.python,
        "scripts/ablation_suite.py",
        "--days",
        str(int(args.ablation_days)),
        "--stride_hours",
        str(int(args.ablation_stride_hours)),
        "--tag",
        f"{args.tag}_abl",
        "--include_no_context",
    ]
    if args.no_telegram:
        ablation_cmd.append("--no_telegram")

    run_id = f"{args.tag}_{_ts()}"
    out_json = os.path.join("analysis", f"optuna_full_run_{run_id}.json")

    if args.detach:
        argv = [args.python, "scripts/optuna_full_run.py"] + [
            "--python",
            args.python,
            "--tag",
            args.tag,
            "--optuna_trials",
            str(int(args.optuna_trials)),
            "--optuna_timeout_sec",
            str(int(args.optuna_timeout_sec)),
            "--optuna_storage",
            str(args.optuna_storage),
            "--optuna_study_name",
            str(args.optuna_study_name),
            "--backtest_days",
            str(int(args.backtest_days)),
            "--backtest_stride_hours",
            str(int(args.backtest_stride_hours)),
        ]
        if args.run_ablation:
            argv.append("--run_ablation")
            argv += ["--ablation_days", str(int(args.ablation_days)), "--ablation_stride_hours", str(int(args.ablation_stride_hours))]
        if args.no_telegram:
            argv.append("--no_telegram")
        info = _detach(argv, env=env, tag=str(args.tag))
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(0)

    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "tag": str(args.tag),
        "pinned_end_time": pinned_end_time,
        "optuna": {
            "trials": int(args.optuna_trials),
            "timeout_sec": int(args.optuna_timeout_sec),
            "storage": str(args.optuna_storage),
            "study_name": str(args.optuna_study_name),
        },
        "steps": [],
    }

    rc_train = _run(train_cmd, env=env)
    result["steps"].append({"name": "train_optuna", "rc": int(rc_train), "cmd": train_cmd})

    rc_eval = _run(eval_cmd, env=env)
    result["steps"].append({"name": "eval_suite", "rc": int(rc_eval), "cmd": eval_cmd})

    if args.run_ablation:
        rc_abl = _run(ablation_cmd, env=env)
        result["steps"].append({"name": "ablation_suite", "rc": int(rc_abl), "cmd": ablation_cmd})

    _write_json(out_json, result)
    raise SystemExit(0 if rc_train == 0 else 2)


if __name__ == "__main__":
    main()

