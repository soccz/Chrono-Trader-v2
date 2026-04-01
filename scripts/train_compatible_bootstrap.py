import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.config import config
from utils.logger import logger
from training import trainer


def _backup_existing_artifacts(tag: str) -> str:
    backup_dir = os.path.join("models", f"bootstrap_backup_{tag}")
    os.makedirs(backup_dir, exist_ok=True)

    for name in os.listdir("models"):
        src = os.path.join("models", name)
        if not os.path.isfile(src):
            continue
        if name == "model_metadata.json" or (
            name.startswith("model_") and name.endswith(".pth")
        ):
            shutil.copy2(src, os.path.join(backup_dir, name))
    return backup_dir


def _parse_model_ids(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"invalid model id: {value}")
        out.append(value)
    return out


def _prepare_subset_ensemble_config(model_ids: List[int]) -> tuple[str | None, str | None]:
    if not model_ids:
        return None, None

    ensemble_path = os.path.join("models", "ensemble_configs.json")
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(ensemble_path)

    with open(ensemble_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    selected = [item for item in payload.get("models", []) if int(item.get("id", -1)) in set(model_ids)]
    if len(selected) != len(model_ids):
        found_ids = sorted(int(item.get("id", -1)) for item in selected)
        raise ValueError(f"requested model_ids={model_ids}, found={found_ids}")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join("models", f"ensemble_configs.bootstrap_backup_{tag}.json")
    temp_path = os.path.join("models", f"ensemble_configs.bootstrap_subset_{tag}.json")
    shutil.copy2(ensemble_path, backup_path)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"models": selected}, f, ensure_ascii=False, indent=2)
    shutil.copy2(temp_path, ensemble_path)
    return backup_path, temp_path


def _restore_subset_ensemble_config(backup_path: str | None, temp_path: str | None) -> None:
    ensemble_path = os.path.join("models", "ensemble_configs.json")
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, ensemble_path)
        os.remove(backup_path)
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bootstrap a runtime-compatible checkpoint on a small market subset."
    )
    ap.add_argument(
        "--markets",
        default="KRW-XRP",
        help="Comma-separated market list for bootstrap fine-tuning.",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--n_models", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--model_ids",
        default="",
        help="Optional comma-separated ensemble model IDs to train, e.g. 4,5.",
    )
    ap.add_argument("--skip_backup", action="store_true")
    args = ap.parse_args()

    markets = [m.strip() for m in str(args.markets).split(",") if m.strip()]
    if not markets:
        raise SystemExit("no markets provided")
    model_ids = _parse_model_ids(args.model_ids)

    backup_dir = None
    if not args.skip_backup:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = _backup_existing_artifacts(tag)

    config.Gan.N_ENSEMBLE_MODELS = max(1, int(args.n_models))
    config.Gan.BATCH_SIZE = max(8, int(args.batch_size))
    config.Gan.EPOCHS = max(1, int(args.epochs))

    logger.info(
        "[BootstrapTrain] starting runtime-compatible bootstrap "
        f"markets={markets} epochs={config.Gan.EPOCHS} "
        f"n_models={config.Gan.N_ENSEMBLE_MODELS} batch_size={config.Gan.BATCH_SIZE}"
    )
    if model_ids:
        logger.info(f"[BootstrapTrain] targeted model_ids={model_ids}")
    if backup_dir:
        logger.info(f"[BootstrapTrain] backup_dir={backup_dir}")

    ensemble_backup_path = None
    ensemble_temp_path = None
    try:
        if model_ids:
            ensemble_backup_path, ensemble_temp_path = _prepare_subset_ensemble_config(model_ids)
            config.Gan.N_ENSEMBLE_MODELS = len(model_ids)
        trainer.run(markets=markets, epochs=config.Gan.EPOCHS)
    finally:
        _restore_subset_ensemble_config(ensemble_backup_path, ensemble_temp_path)

    summary = {
        "ts": datetime.now().isoformat(),
        "markets": markets,
        "epochs": int(config.Gan.EPOCHS),
        "n_models": int(config.Gan.N_ENSEMBLE_MODELS),
        "batch_size": int(config.Gan.BATCH_SIZE),
        "model_ids": model_ids,
        "backup_dir": backup_dir,
    }
    os.makedirs("analysis", exist_ok=True)
    path = os.path.join("analysis", "bootstrap_train_last.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"[BootstrapTrain] summary written to {path}")


if __name__ == "__main__":
    main()
