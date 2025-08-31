# MLOps – 2‑Day (14h) Course

This repository contains the slides, labs, and project scaffold for a 2‑day MLOps training (total 14h):
- **Lectures**: 4h
- **Labs**: 5h
- **Project**: 5h

## Structure
```
mlops_2day_course/
├── slides/
│   └── MLOps_SupdeVinci.pptx
├── labs/
│   ├── day1_hf_sentiment/
│   └── day2_mlflow_tracking/
├── project/
│   └── ml_microservice/
├── docs/
│   └── syllabus.md
└── .github/workflows/
    └── ci.yml
```

## Quickstart
```bash
# create & activate env
python3 -m venv .venv && source .venv/bin/activate
pip install -r project/ml_microservice/requirements.txt

# run microservice locally
make -C project/ml_microservice run

# run tests
make -C project/ml_microservice test
```
