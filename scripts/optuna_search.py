#!/usr/bin/env python
r"""Optuna hyperparameter search for RD-JEPA with MLflow tracking.

Runs ``cfg.optuna_n_trials`` Optuna trials, each training a full model
with sampled hyperparameters. MLflow tracks every trial's params + metrics.
"""
from __future__ import annotations

import argparse
import math

import optuna
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from rd_jepa.config import Config
from rd_jepa.data.forecasting import build_dataloaders
from rd_jepa.eval.forecast_probe import (
    ForecastProbe,
    evaluate_forecast_probe,
    train_forecast_probe,
)
from rd_jepa.models.rd_jepa import RDJEPA
from rd_jepa.train import _evaluate_loop, train_step


def _get_lr_scheduler(optimizer, cfg, total_steps):
    warmup = cfg.lr_warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        if cfg.lr_cosine:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def objective(trial: optuna.Trial, cfg: Config, mlflow_enabled: bool) -> float:
    import mlflow

    # Sample hyperparameters
    cfg.latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
    cfg.n_modes = trial.suggest_int("n_modes", 16, 64, step=16)
    cfg.K_steps = trial.suggest_int("K_steps", 3, 12)
    cfg.dt = trial.suggest_float("dt", 0.05, 0.3)
    cfg.coupling_sparsity = trial.suggest_float("coupling_sparsity", 0.0, 0.8)
    cfg.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    cfg.batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    cfg.vicreg_var_weight = trial.suggest_float("vicreg_var_weight", 0.5, 2.0)
    cfg.vicreg_cov_weight = trial.suggest_float("vicreg_cov_weight", 0.5, 2.0)
    cfg.phase_div_weight = trial.suggest_float("phase_div_weight", 0.1, 2.0)
    cfg.encoder_hidden = trial.suggest_categorical("encoder_hidden", [256, 512, 1024])

    # Run name for MLflow
    run_name = f"trial_{trial.number}"

    mlflow_ctx = (
        mlflow.start_run(run_name=run_name)
        if mlflow_enabled
        else _NullContext()
    )

    with mlflow_ctx:
        if mlflow_enabled:
            mlflow.log_params(cfg.to_dict())

        torch.manual_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loaders = build_dataloaders(cfg)
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        model = RDJEPA(cfg).to(device)
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = cfg.epochs * len(train_loader)
        scheduler = _get_lr_scheduler(optimizer, cfg, total_steps)

        step = 0
        for epoch in range(cfg.epochs):
            model.train()
            for batch in train_loader:
                train_step(model, optimizer, None, batch, cfg, step)
                scheduler.step()
                step += 1

            if (epoch + 1) % cfg.eval_every_n_epochs == 0 or epoch == cfg.epochs - 1:
                val_metrics = _evaluate_loop(model, val_loader, cfg, device)
                if mlflow_enabled:
                    for k, v in val_metrics.items():
                        mlflow.log_metric(k, v, step=epoch)

        # Final probe evaluation
        probe = ForecastProbe(cfg.latent_dim, cfg.horizon, cfg.n_features)
        train_forecast_probe(model, probe, train_loader, cfg, device)
        probe_metrics = evaluate_forecast_probe(model, probe, val_loader, cfg, device)

        if mlflow_enabled:
            mlflow.log_metrics(probe_metrics)

        # Objective: minimize probe MSE on validation
        return probe_metrics["probe/mse"]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna + MLflow search for RD-JEPA")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--exp-name", type=str, default="optuna_search")
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        fast=args.fast,
        exp_name=args.exp_name,
        optuna_n_trials=args.n_trials,
        optuna_timeout=args.timeout,
        mlflow_tracking_uri=args.mlflow_uri,
    )

    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.exp_name)
        mlflow_enabled = True
        print(f"MLflow tracking at: {cfg.mlflow_tracking_uri}")
    except ImportError:
        mlflow_enabled = False
        print("MLflow not installed — running Optuna without MLflow tracking.")

    study = optuna.create_study(
        direction="minimize",
        study_name=cfg.exp_name,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    def obj_wrapper(trial):
        from copy import deepcopy
        trial_cfg = deepcopy(cfg)
        trial_cfg.epochs = cfg.epochs
        trial_cfg.fast = cfg.fast
        return objective(trial, trial_cfg, mlflow_enabled)

    study.optimize(
        obj_wrapper,
        n_trials=cfg.optuna_n_trials,
        timeout=cfg.optuna_timeout,
    )

    print("\n=== Best trial ===")
    print(f"  Value (probe MSE): {study.best_trial.value:.6f}")
    print(f"  Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
