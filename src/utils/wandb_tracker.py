"""
Weights & Biases integration for experiment tracking and model artifacts.

Enable in config.yaml under ``wandb.enabled``. Requires ``wandb`` and
``WANDB_API_KEY`` (or ``wandb login``) when enabled.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from src.config.loader import get_config


def _wandb_settings() -> dict[str, Any] | None:
    """Return wandb config dict if tracking is enabled, else None."""
    config = get_config()
    settings = config.get("wandb") or {}
    if not settings.get("enabled", False):
        return None
    return settings


def _build_run_name(model_name: str, job_type: str) -> str:
    """W&B run name from the model key and pipeline step (e.g. logistic_regression-train)."""
    return f"{model_name}-{job_type}"


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        logging.warning(
            "wandb is enabled in config but not installed; "
            "pip install wandb or set wandb.enabled: false"
        )
        raise exc
    return wandb


def log_training_run(
    *,
    model_name: str,
    metrics: dict[str, Any],
    artifact_path: str,
    best_params: dict[str, Any] | None = None,
    reports_dir: str | None = None,
    used_grid_search: bool = False,
) -> str | None:
    """
    Log a training run: hyperparameters, metrics, model artifact, optional reports.

    Returns the W&B run URL when logging succeeds, else None.
    """
    settings = _wandb_settings()
    if settings is None:
        return None

    wandb = _import_wandb()
    config: dict[str, Any] = cast(dict[str, Any], get_config())
    model_cfg: dict[str, Any] = config["model"]

    run_config = {
        "model_name": model_name,
        "model.active": model_cfg["active"],
        "model.test_size": model_cfg["test_size"],
        "model.random_state": model_cfg["random_state"],
        "model.cv_folds": model_cfg["cv_folds"],
        "model.scoring": model_cfg["scoring"],
        "used_grid_search": used_grid_search,
    }
    if best_params:
        run_config["best_params"] = best_params

    tags = list(settings.get("tags") or [])
    if model_name not in tags:
        tags.append(model_name)

    run = wandb.init(
        project=settings.get("project", "network-monitoring"),
        entity=settings.get("entity"),
        name=_build_run_name(model_name, "train"),
        job_type="train",
        tags=tags,
        config=run_config,
        reinit=True,
    )

    log_metrics = {k: v for k, v in metrics.items() if v is not None}
    if log_metrics:
        wandb.log(log_metrics)

    if settings.get("log_artifact", True) and os.path.isfile(artifact_path):
        model_artifact = wandb.Artifact(
            name=f"{model_name}-model",
            type="model",
            description=f"Trained {model_name} pipeline artifact (joblib)",
            metadata={
                "model_name": model_name,
                "best_params": best_params,
            },
        )
        model_artifact.add_file(artifact_path, name=os.path.basename(artifact_path))
        run.log_artifact(model_artifact)
        logging.info("Logged model artifact to W&B: %s", artifact_path)

    if settings.get("log_grid_search", True) and reports_dir:
        grid_path = os.path.join(reports_dir, "grid_search_results.csv")
        if os.path.isfile(grid_path):
            table_artifact = wandb.Artifact(
                name=f"{model_name}-grid-search",
                type="dataset",
                description="GridSearchCV cv_results export",
            )
            table_artifact.add_file(grid_path)
            run.log_artifact(table_artifact)

    run_url = run.url
    run.finish()
    logging.info("W&B training run logged: %s", run_url)
    return run_url


def log_evaluation_run(
    *,
    model_name: str,
    metrics: dict[str, Any],
    artifact_path: str | None = None,
) -> str | None:
    """
    Log evaluation metrics (and optionally link the model artifact).

    Returns the W&B run URL when logging succeeds, else None.
    """
    settings = _wandb_settings()
    if settings is None:
        return None

    wandb = _import_wandb()

    tags = list(settings.get("tags") or [])
    tags.append("evaluation")
    if model_name not in tags:
        tags.append(model_name)

    run = wandb.init(
        project=settings.get("project", "network-monitoring"),
        entity=settings.get("entity"),
        name=_build_run_name(model_name, "evaluate"),
        job_type="evaluate",
        tags=tags,
        config={"model_name": model_name},
        reinit=True,
    )

    log_metrics = {k: v for k, v in metrics.items() if v is not None}
    if log_metrics:
        wandb.log(log_metrics)

    if (
        settings.get("log_artifact", True)
        and artifact_path
        and os.path.isfile(artifact_path)
    ):
        eval_artifact = wandb.Artifact(
            name=f"{model_name}-eval-snapshot",
            type="model",
            description="Model artifact used for this evaluation",
        )
        eval_artifact.add_file(artifact_path, name=os.path.basename(artifact_path))
        run.log_artifact(eval_artifact)

    run_url = run.url
    run.finish()
    logging.info("W&B evaluation run logged: %s", run_url)
    return run_url
