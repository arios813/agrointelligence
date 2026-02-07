# Cattle Breeding Analytics

This project ingests cattle breeding data, performs EDA, and trains regression models to predict final animal weight.

Usage (example):

```bash
pip install -r requirements.txt
python src/run_pipeline.py --data "/Users/andrewrios/Downloads/pukavy_bulls.csv"
```

Outputs:
- `outputs/predictions.csv` — predictions and true weights
- `models/` — saved models (`model_rf.joblib`, `model_lr.joblib`)
- `outputs/figures/` — EDA and model figures

Project layout:
- `data/` — CSVs (optional)
- `src/` — modular code
- `models/`, `outputs/`
