import matplotlib.pyplot as plt

from pathlib import Path
from src.data.loader import load_datasets, SENSOR_COLS, SETTING_COLS

train, test = load_datasets()

# 1) How imbalanced is our failure label?
print("\nFailure class share in training data: ")
print(train["failure"].value_counts(normalize=True).round(3))

# 2) How long do engines live?
life = train.groupby("unit")["cycle"].max()
print("\nEngine lifetimes (cycles): ")
print(life.describe().round(1))

# 3) Which sensors never change? (useless for predicting anything)
stds = train[SETTING_COLS + SENSOR_COLS].std()
constant = stds[stds < 1e-6].index.tolist()
print("\nConstant columns (we'll drop these later): ", constant)

# 4) Do healthy and dying engines look different?
#    Pick a few sensors and plot them against RUL for one engine.
engine = train[train["unit"] == 1]
show = ["s2", "s4", "s7", "s11", "s12", "s15"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, s in zip(axes.ravel(), show):
    ax.plot(engine["cycle"], engine[s])
    ax.axvline(engine["cycle"].max() - 30, color="red", ls="--", label="danger zone starts")
    ax.set_title(s)
    ax.set_xlabel("cycle")
axes[0, 0].legend()
plt.suptitle("Engine #1: sensors drift as it wears out")
plt.tight_layout()
Path("artifacts").mkdir(exist_ok=True)
plt.savefig("artifacts/eda_engine1.png", dpi=120)
print("\nSaved plot -> artifacts/eda_engine1.png")
