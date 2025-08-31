import numpy as np
import argparse
import os
from functools import partial
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    save_pickle,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np
from utils.utils import parallel_process


def process_single_scan(scan_info, preprocess_config, target_dir):
    """
    Process a single scan for pretraining data.

    Args:
        scan_info: A tuple containing (subject_name, session_name, scan_file, scan_path)
        preprocess_config: Preprocessing configuration dictionary
        target_dir: Target directory for preprocessed data

    Returns:
        Success message or error message
    """
    subject_name, session_name, scan_file, scan_path = scan_info

    # Extract filename without extension to use as identifier
    scan_name = os.path.splitext(os.path.splitext(scan_file)[0])[0]

    try:
        images, image_props = preprocess_case_for_training_without_label(
            images=[read_file_to_nifti_or_np(scan_path)], **preprocess_config
        )
        image = images[0]

        # Create a unique filename for the preprocessed data
        filename = f"{subject_name}_{session_name}_{scan_name}"
        save_path = join(target_dir, filename)
        np.save(save_path + ".npy", image)
        save_pickle(image_props, save_path + ".pkl")

        return f"Processed {subject_name}/{session_name}/{scan_file}"
    except Exception as e:
        return f"Error processing {subject_name}/{session_name}/{scan_file}: {str(e)}"


def preprocess_pretrain_data(in_path: str, out_path: str, num_workers: int = None):
    """
    Preprocess all pretraining data in parallel.

    Args:
        in_path: Path to the source data directory
        out_path: Path to store preprocessed data
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    target_dir = join(out_path, "FOMO60k")
    ensure_dir_exists(target_dir)

    preprocess_config = {
        "normalization_operation": ["volume_wise_znorm"],
        "crop_to_nonzero": True,
        "target_orientation": "RAS",
        "target_spacing": [1.0, 1.0, 1.0],
        "keep_aspect_ratio_when_using_target_size": False,
        "transpose": [0, 1, 2],
    }

    # Collect all scan paths
    scan_infos = []
    for subject_name in sorted(os.listdir(in_path)):
        subject_dir = os.path.join(in_path, subject_name)
        if not os.path.isdir(subject_dir):
            continue

        for session_name in sorted(os.listdir(subject_dir)):
            session_dir = os.path.join(subject_dir, session_name)
            if not os.path.isdir(session_dir):
                continue

            scan_files = [f for f in os.listdir(session_dir) if f.endswith(".nii.gz")]
            for scan_file in scan_files:
                scan_path = os.path.join(session_dir, scan_file)
                scan_infos.append((subject_name, session_name, scan_file, scan_path))

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_scan, preprocess_config=preprocess_config, target_dir=target_dir
    )

    # Process all scans in parallel using the shared utility function
    parallel_process(process_func, scan_infos, num_workers, desc="Preprocessing scans")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", type=str, required=True, help="Path to pretrain data"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to put preprocessed pretrain data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers to use. Default is CPU count - 1",
    )
    args = parser.parse_args()
    preprocess_pretrain_data(
        in_path=args.in_path, out_path=args.out_path, num_workers=args.num_workers
    )
