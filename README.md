# LightGBM Anomaly Classification *(Archived Prototype)*

> **Status: Archived.** This was an early 2024 prototype for multiclass anomaly classification using LightGBM.
> It has been superseded by **[`ae-gbm-fault-classifier`](https://github.com/Thestarki/ae-gbm-fault-classifier)**, which refactors the full Autoencoder + LightGBM fault-diagnosis workflow into a typed, tested, containerized FastAPI microservice (Poetry, Pydantic, Loguru, Docker, pytest, GitHub Actions).

## Contents (Historical Reference)

```text
src/
├── Runner_GBM.py                  # Training and evaluation entrypoint
├── config/config.py               # Paths and hyperparameters
└── Model/
    ├── ReadData.py
    ├── ProcessData.py
    ├── TrainTestValSplit.py
    ├── BuildDataloader.py
    ├── LightGBModel.py
    └── BuildConfusionMatrix.py
```

Please visit **[`ae-gbm-fault-classifier`](https://github.com/Thestarki/ae-gbm-fault-classifier)** for the current maintained implementation.
