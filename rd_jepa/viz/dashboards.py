r"""Helpers to print launch commands + URLs for the MLflow and Optuna dashboards.

Both scripts (``scripts/train.py`` and ``scripts/optuna_search.py``) call
:func:`print_dashboards` at the end of a run so the user gets the exact
commands to open each dashboard and the URL to visit in a browser.
"""
from __future__ import annotations

import shlex


def mlflow_ui_command(tracking_uri: str, port: int = 5000) -> str:
    return f"mlflow ui --backend-store-uri {shlex.quote(tracking_uri)} --port {port}"


def optuna_dashboard_command(storage: str, port: int = 8080) -> str:
    return f"optuna-dashboard {shlex.quote(storage)} --port {port}"


def print_dashboards(
    *,
    mlflow_uri: str | None = None,
    optuna_storage: str | None = None,
    mlflow_port: int = 5000,
    optuna_port: int = 8080,
) -> None:
    """Print the launch commands and browser URLs for the available dashboards."""
    print("\n" + "=" * 70)
    print("Dashboards — run these in a separate terminal, then open the URL:")
    print("-" * 70)
    if mlflow_uri is not None:
        print(f"  MLflow UI         {mlflow_ui_command(mlflow_uri, mlflow_port)}")
        print(f"                    → http://localhost:{mlflow_port}")
    if optuna_storage is not None:
        print(
            f"  Optuna dashboard   {optuna_dashboard_command(optuna_storage, optuna_port)}"
        )
        print(f"                    → http://localhost:{optuna_port}")
    print("=" * 70 + "\n")
