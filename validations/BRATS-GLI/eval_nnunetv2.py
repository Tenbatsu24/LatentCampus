import gc
import time
import json

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_files_list


def eval_runner_brats_gli():
    classes = [1, 2, 3, 4]
    current_dir = Path(__file__).parent
    gt_dir = Path("gt")

    models = [d for d in current_dir.iterdir() if d.is_dir() and d.name != "gt"]

    for model in models:
        model_name = model.name

        if Path(f"{model_name}_metrics.json").exists():
            print(f"Metrics file for {model_name} already exists, skipping...")
            continue

        print(f"Evaluating model: {model_name}")

        pred_files = []
        gt_files = []

        for pred_file_path in (model / "validation").iterdir():
            pred_file_name = pred_file_path.name.replace(".nii.gz", "")
            gt_file_path = (
                    gt_dir
                    / f"{pred_file_name}"
                    / f"{pred_file_name}-seg.nii.gz"
            )

            if not gt_file_path.exists():
                print(f"Ground truth file not found for {pred_file_name}, skipping...")
                continue

            pred_files.append(str(pred_file_path.resolve()))
            gt_files.append(str(gt_file_path.resolve()))

        compute_metrics_on_files_list(
            files_pred=pred_files,
            files_ref=gt_files,
            ignore_label=None,
            image_reader_writer=SimpleITKIO(),
            num_processes=8,
            output_file=f"{model_name}_metrics.json",
            regions_or_labels=classes,
        )
        # Free memory
        gc.collect()
        time.sleep(0.1)

        print(f"Finished evaluating model: {model_name}\n")

    # Get all the *_metrics.json files in the current directory
    metrics_files = list(current_dir.glob("*_metrics.json"))
    all_metrics = []

    # Parse each metrics file
    for file_path in metrics_files:
        model_name = file_path.stem.replace("_metrics", "")
        with open(file_path, "r") as f:
            data = json.load(f)

        class_metrics = data.get("mean", {})
        global_metrics = data.get("foreground_mean", {})

        # Collect all metric names (assuming they are the same across classes)
        metric_names = next(iter(class_metrics.values())).keys() if class_metrics else global_metrics.keys()

        for metric in metric_names:
            row = {
                "type": "CNN" if "cnn" in model_name.lower() else "EVA",
                "model_name": model_name.replace("CNN_", "").replace("EVA_", "").replace("Eva_", ""),
                "metric": metric,
                **{cls: class_metrics.get(str(cls), {}).get(metric, None) for cls in classes},
                "foreground_mean": global_metrics.get(metric, None)
            }
            all_metrics.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    # Save to CSV
    metrics_csv = current_dir / "brats_gli_metrics.csv"

    df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved to {metrics_csv}")


if __name__ == "__main__":
    eval_runner_brats_gli()
