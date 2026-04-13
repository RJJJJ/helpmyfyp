import csv
from pathlib import Path
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerCSVLogger(nnUNetTrainer):
    """
    每個 epoch 結束時，將常見訓練指標輸出到 CSV。
    輸出位置: <output_folder>/epoch_metrics.csv
    """
    # 【終極修正】不使用 kwargs，精準對齊官方傳遞的 5 個參數
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device)
        self.csv_path = None

    def on_train_start(self):
        super().on_train_start()
        self.csv_path = Path(self.output_folder) / "epoch_metrics.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "mean_fg_dice",
                    "ema_fg_dice"
                ])

    def _safe_last(self, obj, key):
        try:
            value = obj.get(key, [])
            if isinstance(value, list) and len(value) > 0:
                return value[-1]
        except Exception:
            pass
        return ""

    def on_epoch_end(self):
        super().on_epoch_end()
        log_store = getattr(self.logger, "my_fantastic_logging", {})
        row = [
            getattr(self, "current_epoch", ""),
            self._safe_last(log_store, "train_losses"),
            self._safe_last(log_store, "val_losses"),
            self._safe_last(log_store, "mean_fg_dice"),
            self._safe_last(log_store, "ema_fg_dice"),
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
