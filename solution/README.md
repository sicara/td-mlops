# Useful commands

## Run MLFlow Tracking Server

```bash
mlflow server
```

## Run training script

```bash
python train_model.py <run_name>
```

To see script usage, run:

```bash
python train_model.py --help
```

## Run model serving API

```bash
uvicorn serving:app --reload
```

## Request model serving API

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@data/test_image.jpg"
```
