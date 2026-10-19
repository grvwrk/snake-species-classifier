from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.models import Inception_V3_Weights, inception_v3


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = APP_DIR / "inception_v3_snakes_aux_best.pth"
DEFAULT_METADATA = APP_DIR / "train.csv"
DEVICE = torch.device("cpu")
TOP_K_DEFAULT = 5


st.set_page_config(
    page_title="Snake Species Classifier",
    page_icon="🐍",
    layout="centered",
)


def normalize_species(value: str) -> str:
    """Normalize species names for joining checkpoint labels to CSV metadata."""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


@st.cache_data(show_spinner=False)
def load_metadata(csv_path: str) -> dict[str, Any]:
    """Load CSV metadata indexed by both scientific name and dataset class_id."""
    path = Path(csv_path)
    if not path.is_file():
        return {"by_species": {}, "by_class_id": {}}

    df = pd.read_csv(path)
    if "binomial" not in df.columns:
        raise ValueError("train.csv must contain a 'binomial' column.")

    detail_columns = [
        "binomial", "snake_sub_family", "genus", "family",
        "poisonous", "country", "continent", "class_id",
    ]
    available = [column for column in detail_columns if column in df.columns]
    by_species: dict[str, dict[str, Any]] = {}
    by_class_id: dict[str, dict[str, Any]] = {}

    # A species may appear in many rows. Aggregate unique metadata values.
    for species, group in df[available].groupby("binomial", dropna=True, sort=False):
        record: dict[str, Any] = {"binomial": str(species).strip()}
        for column in available:
            values = []
            for value in group[column].tolist():
                if pd.isna(value):
                    continue
                value = str(value).strip()
                if not value or value.casefold() in {"unknown", "nan", "none"}:
                    continue
                if value not in values:
                    values.append(value)

            if values:
                if column in {"country", "continent"}:
                    record[column] = values[:8]
                else:
                    record[column] = values[0]

        by_species[normalize_species(str(species))] = record

        # class_id is dataset metadata; use it only to join the predicted class
        # label to CSV, not as a substitute for the model's output index.
        if "class_id" in record:
            try:
                by_class_id[str(int(float(record["class_id"])))] = record
            except (TypeError, ValueError):
                by_class_id[str(record["class_id"])] = record

        alternate = record.get("snake_sub_family")
        if alternate:
            by_species.setdefault(normalize_species(alternate), record)

    return {"by_species": by_species, "by_class_id": by_class_id}

@st.cache_resource(show_spinner="Loading Inception V3 checkpoint...")
def load_model(checkpoint_path: str):
    """Load the saved Inception V3 model and preserve its trained class mapping."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Place inception_v3_snakes_aux_best.pth beside app.py."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    required = {"model_state_dict", "class_to_idx", "num_classes"}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")

    num_classes = int(checkpoint["num_classes"])
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}

    if len(class_to_idx) != num_classes:
        raise ValueError(
            f"Checkpoint says num_classes={num_classes}, "
            f"but class_to_idx contains {len(class_to_idx)} entries."
        )
    if set(class_to_idx.values()) != set(range(num_classes)):
        raise ValueError(
            "class_to_idx values are not contiguous indices from 0 to "
            f"{num_classes - 1}. Check the checkpoint's saved class mapping."
        )

    # Resolve the saved mapping's orientation robustly. The intended form is
    # output_index -> class_label, but some training notebooks serialize the
    # class_to_idx mapping under the idx_to_species name by mistake.
    # Never silently interpret the neural-network output index as CSV class_id.
    raw_idx_to_species = checkpoint.get("idx_to_species")
    idx_to_species: dict[int, str] = {}

    if isinstance(raw_idx_to_species, dict) and raw_idx_to_species:
        # First try the intended index -> label orientation.
        candidate = {}
        for key, value in raw_idx_to_species.items():
            try:
                candidate[int(key)] = str(value)
            except (TypeError, ValueError):
                candidate = {}
                break

        expected_indices = set(range(num_classes))
        if expected_indices.issubset(candidate):
            idx_to_species = candidate
        else:
            # If keys are class labels and values are output indices, invert
            # this mapping only when the values unambiguously cover all indices.
            inverse = {}
            try:
                for key, value in raw_idx_to_species.items():
                    index = int(value)
                    inverse[index] = str(key)
            except (TypeError, ValueError):
                inverse = {}

            if expected_indices.issubset(inverse):
                idx_to_species = inverse

    if not idx_to_species:
        # class_to_idx is explicitly label -> output index. Invert it only
        # after confirming it covers every output index.
        idx_to_species = {index: label for label, index in class_to_idx.items()}

    missing_indices = set(range(num_classes)) - set(idx_to_species)
    if missing_indices:
        raise ValueError(
            "Neither saved class mapping covers every model output index. "
            f"Missing indices: {sorted(missing_indices)[:10]}. "
            f"class_to_idx sample: {list(class_to_idx.items())[:5]}; "
            f"idx_to_species sample: "
            f"{list(raw_idx_to_species.items())[:5] if isinstance(raw_idx_to_species, dict) else raw_idx_to_species!r}"
        )

    model = inception_v3(
        weights=None,
        aux_logits=True,
        num_classes=1000,
        transform_input=False,
        init_weights=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if model.AuxLogits is None:
        raise RuntimeError("Inception V3 auxiliary classifier was not created.")
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    preprocess = Inception_V3_Weights.DEFAULT.transforms()
    return model, preprocess, idx_to_species, num_classes, checkpoint


def metadata_for_prediction(class_label: str, metadata: dict[str, Any]):
    """Resolve a checkpoint class label to the CSV record.

    A checkpoint label may be a scientific name or a dataset class_id.
    This lookup never treats the neural-network output index itself as class_id.
    """
    by_species = metadata.get("by_species", {})
    by_class_id = metadata.get("by_class_id", {})

    species_record = by_species.get(normalize_species(class_label))
    if species_record:
        return species_record

    # Numeric labels are dataset class IDs only if the checkpoint mapping
    # explicitly maps the predicted output index to that numeric class label.
    label = str(class_label).strip()
    try:
        class_id = str(int(float(label)))
    except (TypeError, ValueError):
        return None
    return by_class_id.get(class_id)


def display_metadata(record: dict[str, Any] | None):
    if not record:
        st.info(
            "No matching species record was found in train.csv. "
            "Check that train.csv is beside app.py and that its species names "
            "match the checkpoint's class labels."
        )
        return

    left, right = st.columns(2)
    with left:
        if record.get("binomial"):
            st.markdown("**Scientific name**")
            st.write(record["binomial"])
        if record.get("genus"):
            st.markdown("**Genus**")
            st.write(record["genus"])
        if record.get("family"):
            st.markdown("**Family**")
            st.write(record["family"])

    with right:
        poisonous = record.get("poisonous")
        if poisonous is not None:
            st.markdown("**Dataset venom label**")
            try:
                poisonous_value = int(float(poisonous))
                if poisonous_value == 1:
                    st.error("Marked poisonous in dataset")
                elif poisonous_value == 0:
                    st.success("Marked non-poisonous in dataset")
                else:
                    st.write(str(poisonous))
            except (TypeError, ValueError):
                st.write(str(poisonous))
        if record.get("snake_sub_family"):
            st.markdown("**Dataset sub-family field**")
            st.write(record["snake_sub_family"])

    countries = record.get("country", [])
    continents = record.get("continent", [])
    if countries:
        st.markdown("**Countries recorded in dataset**")
        st.write(", ".join(countries))
    if continents:
        st.markdown("**Continents recorded in dataset**")
        st.write(", ".join(continents))

    st.caption(
        "Taxonomy, distribution, and the poisonous label are dataset metadata, "
        "not a field-verified identification or a guarantee of safety."
    )


def predict(
    image: Image.Image,
    model,
    preprocess,
    idx_to_species: dict[int, str],
    num_classes: int,
    top_k: int,
) -> list[dict[str, Any]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(top_k, num_classes))

    results = []
    for probability, index in zip(values.tolist(), indices.tolist()):
        species = idx_to_species.get(int(index), f"Unknown class {index}")
        results.append({
            "class_index": int(index),
            "species": species,
            "confidence": float(probability),
            "confidence_percent": float(probability * 100.0),
        })
    return results


def main():
    st.title("Snake Species Classifier")
    st.caption("Inception V3 · CPU inference")
    st.write(
        "Upload an image to see the model's top predicted species and available "
        "dataset details. This is an educational classifier, not a reliable "
        "method for deciding whether a snake is safe to approach."
    )

    if not DEFAULT_CHECKPOINT.exists():
        st.error(
            "Checkpoint not found. Copy `inception_v3_snakes_aux_best.pth` "
            "into the same folder as `app.py`."
        )
        st.code(str(DEFAULT_CHECKPOINT))
        st.stop()

    try:
        model, preprocess, idx_to_species, num_classes, checkpoint = load_model(
            str(DEFAULT_CHECKPOINT)
        )
        metadata = load_metadata(str(DEFAULT_METADATA))
    except Exception as exc:
        st.error(f"Could not load model or metadata: {exc}")
        st.stop()

    with st.expander("Model details"):
        st.write(f"Architecture: `{checkpoint.get('architecture', 'inception_v3')}`")
        st.write(f"Classes: `{num_classes}`")
        st.write("Saved validation accuracy: `87.04%`")
        st.write("Device: `CPU`")
        st.write(
            "Species metadata: "
            + ("`train.csv` loaded" if metadata.get("by_species") else "`train.csv` not found or empty")
        )

    uploaded = st.file_uploader(
        "Choose a snake image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
    )
    top_k = st.slider(
        "Number of predictions",
        min_value=1,
        max_value=min(10, num_classes),
        value=min(TOP_K_DEFAULT, num_classes),
    )

    if uploaded is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read that image: {exc}")
            st.stop()

        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify image", type="primary", use_container_width=True):
            with st.spinner("Running CPU inference..."):
                try:
                    predictions = predict(
                        image, model, preprocess, idx_to_species, num_classes, top_k
                    )
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    st.stop()

            st.subheader("Top predictions")
            for rank, item in enumerate(predictions, start=1):
                # The checkpoint may map an output to a numeric dataset class_id
                # (for example, "462"). Resolve that label to train.csv first,
                # then show the scientific binomial as the visible prediction.
                record = metadata_for_prediction(item["species"], metadata)
                display_name = (
                    str(record["binomial"]).strip()
                    if record and record.get("binomial")
                    else item["species"]
                )

                st.markdown(f"### {rank}. {display_name}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(min(max(item["confidence"], 0.0), 1.0))
                with col2:
                    st.metric("Confidence", f"{item['confidence_percent']:.2f}%")

                with st.expander(f"Details: {display_name}", expanded=(rank == 1)):
                    display_metadata(record)

            st.caption(
                "Confidence is a softmax score, not a calibrated probability that "
                "the identification is correct. The poisonous label reflects the "
                "dataset only. Never approach or handle a snake based on this output."
            )


if __name__ == "__main__":
    main()
