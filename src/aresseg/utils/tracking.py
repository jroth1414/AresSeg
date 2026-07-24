"""Optional experiment-tracker context manager (MLflow / W&B) that no-ops when disabled.

MLflow and W&B are NOT in requirements.txt. Imports are guarded; if unavailable or disabled
in config, ``tracker`` is a no-op. Install ``mlflow`` only if ``tracking.mlflow: true``.
"""

from contextlib import contextmanager


@contextmanager
def tracker(cfg: dict, run_name: str):
    use_mlflow = cfg.get("tracking", {}).get("mlflow", False)
    use_wandb = cfg.get("tracking", {}).get("wandb", False)
    handle = None
    try:
        if use_mlflow:
            import mlflow

            mlflow.start_run(run_name=run_name)
            handle = ("mlflow", mlflow)
        elif use_wandb:
            import wandb

            handle = ("wandb", wandb.init(name=run_name, mode="offline"))
        yield handle
    finally:
        if handle and handle[0] == "mlflow":
            handle[1].end_run()
        elif handle and handle[0] == "wandb":
            handle[1].finish()
