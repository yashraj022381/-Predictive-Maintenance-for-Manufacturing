"""Rebuild every artifact from the raw data, in the right order."""
import subprocess
import sys

STEPS = [
    "src.features.engineering",   # features + train/val/test split
    "src.models.anomaly",         # Isolation Forest
    "src.models.autoencoder",     # Autoencoder
    "src.models.failure_clf",     # Random Forest + XGBoost risk
    "src.models.rul_models",      # XGBoost + LSTM remaining life
    "src.explain.shap_explainer", # SHAP plots
    "src.genai.rag",              # manual index for the AI
]
for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, "-m", step], check=True)
print("\nAll artifacts rebuilt.")
