# Snake Species Classifier — Minimal CPU Inference

A minimal Streamlit upload interface for the saved Inception V3 snake-species classifier. It runs on **CPU only** and does not train a model or download pretrained weights.

## Files

```text
├── app.py
├── requirements.txt
└── inception_v3_snakes_aux_best.pth   
```

## 1. Create an environment

Python 3.10 or 3.11 is recommended.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If pip installs a CUDA-enabled PyTorch build on a machine without CUDA, inference still explicitly uses CPU. To install CPU-only wheels and avoid unnecessary CUDA packages, use the official PyTorch CPU wheel index for your operating system/Python version, then install the remaining requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "streamlit>=1.37,<2.0" "Pillow>=10.0"
```

For reproducible deployment, pin the exact tested `torch` and `torchvision` versions together after confirming the checkpoint loads.

## 3. Run the app

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser, upload a JPG/PNG/WebP image, choose the number of predictions, and click **Classify image**.

## Implementation notes

- Uses `inception_v3(weights=None, aux_logits=True, ...)` and loads all weights from your checkpoint. No pretrained weights are downloaded.
- Recreates both the main and auxiliary classifier heads, then loads the checkpoint with `strict=True` so architecture mismatches fail loudly.
- Uses `Inception_V3_Weights.DEFAULT.transforms()` for the standard Inception V3 inference preprocessing (resize, center crop to 299, tensor conversion, ImageNet normalization).
- Calls `model.eval()` and `torch.inference_mode()`. Evaluation returns main-classifier logits; the auxiliary branch is not used for predictions.
- Runs on CPU regardless of whether CUDA is present.
- The displayed softmax score is not a calibrated probability and must not be used as a safety guarantee. Do not handle or approach a snake based on this model's output.

## Troubleshooting

### `Checkpoint not found`
Put `inception_v3_snakes_aux_best.pth` beside `app.py`.

### State-dict mismatch
Make sure you copied the **auxiliary fine-tuning checkpoint** `inception_v3_snakes_aux_best.pth`, not an earlier checkpoint that was saved without the auxiliary branch. The app intentionally uses strict loading to detect mismatched checkpoints.

### Torch / torchvision compatibility error
Install a compatible `torch` and `torchvision` pair. The training environment used PyTorch 2.8.x with torchvision 0.23.x; matching those versions is a sensible first attempt if available for your Python/platform.

### Slow first prediction
CPU inference can be slower than GPU inference, especially on the first run. The model is cached in the Streamlit process after loading.
