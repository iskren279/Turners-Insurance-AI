# TwinCar — Vehicle Make & Model Recognition

Brainster Data Science Academy 2026 · Machine Learning Final Project · Group 2

## Overview

TwinCar is a vehicle recognition project focused on predicting vehicle make and model from car images. Year is included as an auxiliary output derived from the predicted fine-grained class.

The final system is not just a notebook experiment. It includes:

- a reusable Python package
- training and evaluation scripts
- robust evaluation transforms
- batch prediction
- FastAPI inference endpoint
- Gradio Hugging Face demo
- Hugging Face model hosting
- W&B experiment tracking
- CI checks and automated tests

The final deployed model is an EfficientNet-B3 classifier fine-tuned on Stanford Cars. It predicts one of 196 Stanford Cars fine-grained classes, then derives make, model, and year from the predicted class metadata.

## Final status

| Area | Status |
|---|---|
| Main model | EfficientNet-B3 |
| Dataset used for final training | Stanford Cars |
| Input size | 300 px |
| Number of classes | 196 fine-grained classes |
| Final checkpoint | `efficientnet_b3_stanford300_augv2_best.pt` |
| Model hosting | Hugging Face model repo |
| Demo | Hugging Face Space |
| API | FastAPI |
| Batch prediction | Supported |
| Experiment tracking | W&B |
| CI | Ruff + pytest |
| CompCars | Inspected and used for external validation/reconnaissance, not blindly merged |
| YOLO cropper | Optional experimental comparison mode, off by default |

## What the model predicts

The model directly predicts a fine-grained Stanford Cars class, for example:

```text
Dodge Charger SRT-8 2009
```

From that predicted fine class, the system derives:

- **make** — e.g. `Dodge`
- **model** — e.g. `Charger SRT-8`
- **year** — e.g. `2009`

Year is included in the API and Gradio output, but it is not predicted by a separate year-specific head. It is derived from the predicted fine-grained class metadata. This is intentional because year is treated as an auxiliary output.

## Final validation results

Final quantitative comparison is reported on the locked Stanford validation split using the same evaluation protocol for all compared models.

| Transform        | Fine acc | Make acc | Model acc | Year acc | Top-3 acc | Top-5 acc |
| ---------------- | -------: | -------: | --------: | -------: | --------: | --------: |
| clean            |   0.7864 |   0.8692 |    0.7925 |   0.8913 |    0.9196 |    0.9521 |
| robust_light     |   0.7882 |   0.8680 |    0.7944 |   0.8956 |    0.9159 |    0.9490 |
| robust_hard      |   0.6839 |   0.7778 |    0.6900 |   0.8355 |    0.8600 |    0.9055 |
| robust_occlusion |   0.6317 |   0.7317 |    0.6366 |   0.8048 |    0.8060 |    0.8600 |

Compared with the earlier EfficientNet-B3 candidate without augmentation v2, this model improves clean fine accuracy by **+1.6 pts** (`0.770 → 0.786`) and robustness substantially — **robust_hard +14 pts** (`0.543 → 0.684`) and **robust_occlusion +13 pts** (`0.502 → 0.632`).

## Main technical story

The final model was not chosen only by clean validation accuracy. The project followed a diagnosis → fix → measure workflow:

1. Train baseline models.
2. Compare multiple backbones.
3. Identify EfficientNet-B3 as the strongest candidate.
4. Add a stronger car-specific augmentation stack.
5. Evaluate clean and robust performance.
6. Promote the best checkpoint through the Hugging Face model manifest.
7. Smoke test both the Gradio demo and FastAPI endpoint.

The strongest final story is:

```text
EfficientNet-B3 + augmentation v2 improved clean accuracy while also improving robustness under harder perturbations and synthetic occlusion.
```

## Robustness evaluation

The final model was evaluated under multiple transforms:

- **clean** — standard validation preprocessing
- **robust_light** — mild production-like perturbations
- **robust_hard** — stronger blur, lighting, color, and geometric perturbations
- **robust_occlusion** — synthetic occlusion/erasing stress test

These tests do not replace real-world field validation, but they give measurable evidence of how the model behaves under controlled distribution shifts.

## Project structure

```text
src/twincar/       Reusable Python package
scripts/           CLI scripts for data prep, training, evaluation, prediction
configs/           Configuration files
splits/            Train/validation/evaluation manifests
docs/              Experiment notes, decisions, smoke tests, reports
app/               Gradio application
tests/             Automated tests
slides/            Presentation material
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional detector dependencies for YOLO cropper experiments:

```bash
pip install -e ".[detect]"
```

API dependencies:

```bash
pip install -e ".[api]"
```

## Data setup

Copy the dataset template:

```bash
cp configs/datasets.example.yaml configs/datasets.local.yaml
```

Edit:

```text
configs/datasets.local.yaml
```

with your local dataset paths.

The Stanford Cars devkit and images should exist locally. Dataset files and `configs/datasets.local.yaml` must not be committed.

Prepare labels and splits:

```bash
python scripts/prepare_data.py --config configs/datasets.local.yaml
```

This writes:

- `splits/stanford_labels.csv`
- `splits/stanford_train.csv`
- `splits/stanford_val.csv`
- `configs/labels.yaml`

The split uses a fixed seed for reproducibility.

## Training

Baseline/default training:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --data-config configs/datasets.local.yaml
```

Final augmentation-v2 training:

```bash
mkdir -p outputs/checkpoints/efficientnet-b3-augv2-final

WANDB_MODE=online \
WANDB_NAME="efficientnet-b3-augv2-final" \
WANDB_TAGS="augmentation,occlusion,efficientnet-b3,300,final-candidate" \
python scripts/train.py \
  --config configs/base.yaml \
  --data-config configs/datasets.local.yaml \
  --backbone efficientnet_b3 \
  --image-size 300 \
  --epochs 22 \
  --batch-size 16 \
  --lr 0.001 \
  --device cuda \
  --train-aug v2 \
  --output-dir outputs/checkpoints/efficientnet-b3-augv2-final
```

Fast smoke test:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --data-config configs/datasets.local.yaml \
  --smoke-test
```

Training logs include:

- train loss
- validation loss
- fine-class accuracy
- make accuracy
- model accuracy
- year accuracy
- top-3 accuracy
- top-5 accuracy

W&B logging is supported when enabled through the config or environment.

## Evaluation

Evaluate a checkpoint on the clean validation split:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/checkpoints/efficientnet-b3-augv2-final/checkpoints/best.pt \
  --split-csv splits/stanford_val.csv \
  --data-config configs/datasets.local.yaml \
  --transform clean \
  --device cuda \
  --output docs/experiments/efficientnet-b3-augv2-final-clean.md
```

Run all final evaluation transforms:

```bash
for T in clean robust_light robust_hard robust_occlusion; do
  python scripts/evaluate_checkpoint.py \
    --checkpoint outputs/checkpoints/efficientnet-b3-augv2-final/checkpoints/best.pt \
    --split-csv splits/stanford_val.csv \
    --data-config configs/datasets.local.yaml \
    --transform "$T" \
    --device cuda \
    --output "docs/experiments/efficientnet-b3-augv2-final-${T}.md"
done
```

Use `--device cpu` instead of `--device cuda` when running on a machine without a CUDA GPU.

## Batch prediction

Run batch prediction on a folder of images:

```bash
python scripts/predict_batch.py \
  --images path/to/images \
  --output predictions.csv \
  --checkpoint outputs/checkpoints/efficientnet-b3-augv2-final/checkpoints/best.pt \
  --top-k 3 \
  --device cuda
```

Run in stub/CI mode without a checkpoint:

```bash
python scripts/predict_batch.py \
  --images path/to/images \
  --output predictions.csv
```

## FastAPI

Start the API in stub/default mode:

```bash
uvicorn twincar.api:app --reload
```

Start the API from the Hugging Face model repo manifest:

```bash
env -u TWINCAR_CHECKPOINT \
HF_MODEL_REPO=twincar-group2/twincar-classifier \
TWINCAR_DEVICE=cpu \
uvicorn twincar.api:app --host 0.0.0.0 --port 8000 --reload
```

Start the API with a local checkpoint:

```bash
TWINCAR_CHECKPOINT=outputs/checkpoints/efficientnet-b3-augv2-final/checkpoints/best.pt \
TWINCAR_DEVICE=cpu \
uvicorn twincar.api:app --host 0.0.0.0 --port 8000 --reload
```

Important: `TWINCAR_CHECKPOINT` takes priority over `HF_MODEL_REPO`. If `TWINCAR_CHECKPOINT` is set to an old local checkpoint, the API will load that local checkpoint instead of the Hugging Face manifest checkpoint. Clear it with:

```bash
unset TWINCAR_CHECKPOINT
```

### API endpoints

| Method | Path          | Description                                        |
| ------ | ------------- | -------------------------------------------------- |
| GET    | `/health`     | Liveness check with `model_loaded` and `stub_mode` |
| GET    | `/model-info` | Architecture, class count, device, and image size  |
| POST   | `/predict`    | Upload image and return make/model/year/top-k      |

Example:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s http://127.0.0.1:8000/model-info | python -m json.tool

curl -s -X POST \
  -F "file=@path/to/car.jpg" \
  "http://127.0.0.1:8000/predict?top_k=3" | python -m json.tool
```

## Gradio demo

Launch the local Gradio app in stub mode:

```bash
python app/gradio_app.py
```

Launch with a local checkpoint:

```bash
python app/gradio_app.py \
  --checkpoint outputs/checkpoints/efficientnet-b3-augv2-final/checkpoints/best.pt
```

Launch from the Hugging Face model repo manifest:

```bash
env -u TWINCAR_CHECKPOINT \
HF_MODEL_REPO=twincar-group2/twincar-classifier \
TWINCAR_DEVICE=cpu \
python app/gradio_app.py
```

Alternative explicit repo argument:

```bash
python app/gradio_app.py \
  --repo twincar-group2/twincar-classifier \
  --device cpu
```

The Gradio demo supports:

- image upload
- webcam/clipboard input
- official full-image prediction
- make/model/year display
- top-k predictions
- optional YOLO crop comparison mode

## YOLO cropper comparison mode

The demo includes an optional, default-off YOLO cropper comparison mode.

When enabled, it:

1. runs a pretrained YOLO vehicle detector,
2. crops the highest-confidence detected vehicle,
3. classifies the crop with the same EfficientNet-B3 classifier,
4. compares the crop prediction against the official full-image prediction.

Important scope decisions:

- YOLO is not trained in this project.
- It uses pretrained COCO weights only.
- The official/default prediction remains the full-image EfficientNet-B3 prediction.
- FastAPI and batch prediction are unaffected.
- If YOLO, ultralytics, or weights are unavailable, the app falls back gracefully.

The cropper was not made default because the A/B test showed that crop-first can change predictions and reduce confidence on some Stanford images. It is therefore kept as an experimental comparison tool, not a production default.

## Hugging Face deployment

### Model repo

The final model is hosted on Hugging Face:

```text
https://huggingface.co/twincar-group2/twincar-classifier
```

Current final checkpoint:

```text
efficientnet_b3_stanford300_augv2_best.pt
```

The model repo includes:

- final EfficientNet-B3 augmentation-v2 checkpoint
- previous EfficientNet-B3 checkpoint
- older ResNet18 baseline checkpoint
- `checkpoint_manifest.json`

The manifest points the API and Space to the current promoted checkpoint.

### Demo Space

The live demo is hosted on Hugging Face Spaces:

```text
https://huggingface.co/spaces/twincar-group2/twincar-demo
```

The Space loads the model through the Hugging Face model repo manifest.

## W&B tracking

Training and evaluation were tracked in W&B:

```text
https://wandb.ai/hzlatevskii-brainster-data-science-academy/twincar
```

W&B was used to compare:

- ResNet18
- ResNet50
- ConvNeXt-Tiny
- EfficientNet-B3
- EfficientNet-B3 augmentation-v2

## CompCars status

CompCars was inspected and used for external validation/reconnaissance, but it was not blindly merged into final training.

The reason is that Stanford Cars and CompCars have a significant domain and label-distribution gap:

- Stanford Cars is a clean fine-grained benchmark with 196 make/model/year classes.
- CompCars contains different image domains and different make/model taxonomies.
- Exact Stanford Cars ↔ CompCars make/model/year overlap was too small and biased for safe blind merging.
- Make-level external validation on CompCars showed a strong cross-domain performance drop.

This confirmed that CompCars integration is a domain adaptation problem, not a simple data-merge task.

Future work should build a verified Stanford Cars ↔ CompCars alias map, train on a controlled filtered subset, and validate on a true cross-domain holdout.

## Held-out Stanford test status

The local Stanford Cars test images were available and were used for qualitative API/demo smoke testing.

However, the available `cars_test_annos.mat` file contained only bounding boxes and filenames:

```text
bbox_x1
bbox_y1
bbox_x2
bbox_y2
fname
```

It did not include class labels.

The provided Kaggle mirror, `eduardo4jesus/stanford-cars-dataset`, was also checked. It included:

```text
cars_meta.mat
cars_train_annos.mat
cars_test_annos.mat
```

but did not provide:

```text
cars_test_annos_withlabels.mat
cars_annos.mat
```

Final quantitative model comparison is therefore reported on the locked Stanford validation split using an identical protocol and seed for every compared model. This keeps model-to-model deltas valid. A labeled held-out Stanford test evaluation would be a straightforward extension if `cars_test_annos_withlabels.mat` is obtained.

## Intended use

TwinCar is intended for educational and prototype-level vehicle recognition experiments, especially make/model recognition from car images similar to Stanford Cars.

Appropriate uses:

- fine-grained vehicle make/model recognition demo
- model comparison and robustness analysis
- prototype vehicle inspection workflow
- academic/academy project demonstration

## Limitations

- The final model is trained on Stanford Cars, not real drone/robot production footage.
- CompCars showed a strong domain gap and was not blindly merged into final training.
- True top-down drone views remain out-of-distribution.
- Robustness tests use controlled synthetic perturbations, not full real-world field validation.
- Year is derived from fine-grained class metadata, not learned as an independent year model.
- Strong non-car/out-of-distribution rejection is not implemented yet.
- Similar models and years can be confused because fine-grained vehicle classification depends on subtle visual details.

## Future work

- Build a verified Stanford Cars ↔ CompCars alias map.
- Fine-tune on a controlled CompCars surveillance subset.
- Add real production/drone/robot images for target-domain validation.
- Add non-car/out-of-distribution rejection by reusing the YOLO detector as a vehicle gate when no vehicle is detected, plus a softmax-confidence threshold for unknown-car/open-set cases.
- Add detector-assisted preprocessing as a validated default only if it improves real metrics.
- Explore multi-head prediction for make/model/year if independent outputs become necessary.
- Evaluate on labeled held-out Stanford test data if `cars_test_annos_withlabels.mat` becomes available.

## Team

| Role                    | Member    |
| ----------------------- | --------- |
| Tech Lead               | Hristijan |
| Modeling Lead           | Huma      |
| Productization Lead     | Bojan     |
| Documentation Owner     | Marija    |
| Presentation / QA Owner | Zoran     |

## Final presentation

- [PDF version](slides/TwinCar_Presentation.pdf)
- [PPTX version](slides/TwinCar_Presentation.pptx)

## Links

- GitHub repo: [https://github.com/Hristijan-kiko/twincar](https://github.com/Hristijan-kiko/twincar)
- Model card: [`docs/model-card.md`](docs/model-card.md)
- W&B project: [https://wandb.ai/hzlatevskii-brainster-data-science-academy/twincar](https://wandb.ai/hzlatevskii-brainster-data-science-academy/twincar)
- Hugging Face model: [https://huggingface.co/twincar-group2/twincar-classifier](https://huggingface.co/twincar-group2/twincar-classifier)
- Hugging Face Space: [https://huggingface.co/spaces/twincar-group2/twincar-demo](https://huggingface.co/spaces/twincar-group2/twincar-demo)
- YOLO cropper A/B report: [`docs/experiments/yolo-cropper-ab-v1.md`](docs/experiments/yolo-cropper-ab-v1.md)
- Augmentation-v2 report: [`docs/experiments/augmentation-v2.md`](docs/experiments/augmentation-v2.md)
- FastAPI smoke test: [`docs/experiments/api-smoke-v2.md`](docs/experiments/api-smoke-v2.md)
- HF Space smoke test: [`docs/experiments/hf-space-smoke-v2.md`](docs/experiments/hf-space-smoke-v2.md)