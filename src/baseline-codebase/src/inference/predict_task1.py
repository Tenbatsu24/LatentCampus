#!/usr/bin/env python
import argparse
import torch
from torch.nn.functional import softmax
from data.task_configs import task1_config
from inference.predict import predict_from_config, save_output_txt

# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task1_config,
    # Add inference-specific configs
    "model_path": "/app/model.ckpt",  # Path to model (inside container!)
    "patch_size": (96, 96, 96),  # Patch size for inference
}


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on FOMO Task 1 (Infarct Detection)"
    )

    # Input and output paths using modality names from task config
    parser.add_argument(
        "--dwi_b1000", type=str, required=True, help="Path to DWI image (NIfTI format)"
    )
    parser.add_argument(
        "--flair",
        type=str,
        required=True,
        help="Path to T2FLAIR image (NIfTI format)",
    )
    parser.add_argument(
        "--adc", type=str, required=True, help="Path to ADC image (NIfTI format)"
    )
    parser.add_argument(
        "--swi", type=str, required=False, help="Path to SWI image (NIfTI format)"
    )
    parser.add_argument(
        "--t2s", type=str, required=False, help="Path to T2* image (NIfTI format)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for prediction"
    )

    # Parse arguments
    args = parser.parse_args()

    assert (args.swi and not args.t2s) or (
        not args.swi and args.t2s
    ), "Either --swi or --t2s must be provided, but not both."

    # Map arguments to modality paths in expected order from task config
    modality_paths = [args.dwi_b1000, args.flair, args.adc, args.swi or args.t2s]
    output_path = args.output

    # Run prediction using the shared prediction logic
    predictions_original, _ = predict_from_config(
        modality_paths=modality_paths,
        predict_config=predict_config,
    )

    # softmax output to get probability
    probabilities = softmax(predictions_original, dim=1)

    save_output_txt(probabilities[0][1], output_path)


if __name__ == "__main__":
    main()
