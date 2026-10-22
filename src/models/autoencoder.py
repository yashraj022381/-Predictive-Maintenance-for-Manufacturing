"""Anomaly detection, part 2: Autoencoder. Squeeze 15 smooth numbers to 4,
then rebuild 15. Big rebuild error = weird machine."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.anomaly import HEALTHY_RUL, load_splits, report, save_threshold


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, bottleneck: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(),
                                     nn.Linear(32, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 32), nn.ReLU(),
                                     nn.Linear(32, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Average squared rebuild mistake for each row."""
    model.eval()
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        return ((model(t) - t) ** 2).mean(dim=1).numpy()


def run(config_path: str = "configs/config.yaml", epochs: int = 300, patience: int = 20) -> None:
    cfg, train, val, test, cols = load_splits(config_path)
    art = Path(cfg["paths"]["artifacts_dir"])
    seed = cfg["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # STEP 1 (must come first!): choose the columns BEFORE anything else uses them
    window = cfg["features"]["window"]
    cols = [c for c in cols if c.endswith(f"_mean{window}")]
    print(f"Autoencoder uses {len(cols)} smoothed features")

    healthy_tr = train[train["rul"] >= HEALTHY_RUL][cols]
    healthy_val = val[val["rul"] >= HEALTHY_RUL][cols]

    # STEP 2: scale using only those columns
    scaler = StandardScaler().fit(healthy_tr)
    x_tr = torch.tensor(scaler.transform(healthy_tr), dtype=torch.float32)
    x_val = scaler.transform(healthy_val)

    # STEP 3: train
    model = Autoencoder(len(cols), bottleneck=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 256):
            batch = x_tr[perm[i:i + 256]]
            opt.zero_grad()
            loss_fn(model(batch), batch).backward()
            opt.step()

        val_loss = reconstruction_error(model, x_val).mean()
        if val_loss < best - 1e-5:
            best, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}")
                break
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | healthy-val error {val_loss:.4f}")

    model.load_state_dict(best_state)

    # STEP 4: score and judge
    def score(df):
        return reconstruction_error(model, scaler.transform(df[cols]))

    threshold = np.percentile(score(train[train["rul"] >= HEALTHY_RUL]), 99)
    report("Autoencoder | validation", score(val), val, threshold)
    report("Autoencoder | test", score(test), test, threshold)

    # STEP 5: save
    torch.save(model.state_dict(), art / "autoencoder.pt")
    joblib.dump(scaler, art / "ae_scaler.joblib")
    (art / "ae_cols.json").write_text(json.dumps(cols, indent=2))
    save_threshold(art, "autoencoder", threshold)
    print("\nSaved model, scaler, columns and threshold to artifacts/")


if __name__ == "__main__":
    run()
