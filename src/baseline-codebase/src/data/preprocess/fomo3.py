import os
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from data.task_configs import task3_config
from utils.utils import parallel_process


def process_subject(task_info):
    """
    Process a single subject for Task 3.

    Args:
        task_info: A tuple containing (subject, source_path, labels_dir, pp_config, target_preprocessed, prefix)

    Returns:
        Success message or error message
    """
    subject, source_path, labels_dir, pp_config, target_preprocessed, prefix = task_info

    try:
        subject_id = subject.replace("sub_", "")
        session_path = join(source_path, "preprocessed", subject, "ses_1")

        if not os.path.isdir(session_path):
            return f"Error: {session_path} is not a valid directory"

        # Get label file
        label_file = join(labels_dir, subject, "ses_1", "label.txt")
        if not os.path.exists(label_file):
            return f"Error: No label file found for {subject}"

        # Read the age from the label file - keep as continuous value for regression
        with open(label_file, "r") as f:
            try:
                age = float(f.read().strip())
            except ValueError:
                return f"Error: Invalid age format in {label_file}"

        # Get all image files in the session directory
        image_files = [f for f in os.listdir(session_path) if f.endswith(".nii.gz")]

        # There should be two MR images per subject
        if len(image_files) != 2:
            return f"Warning: Expected 2 images for {subject}, but found {len(image_files)}. Still processing."

        # Sort images to ensure consistent ordering
        sorted_image_files = sorted(image_files)

        # Load images for preprocessing
        images = []
        for image_file in sorted_image_files:
            source_img = join(session_path, image_file)
            # Load image
            img = nib.load(source_img)
            images.append(img)

        # Apply preprocessing
        preprocessed_images, _ = preprocess_case_for_training_without_label(
            images=images,
            normalization_operation=[
                pp_config["norm_op"] for _ in pp_config["modalities"]
            ],
            allow_missing_modalities=False,
            crop_to_nonzero=pp_config["crop_to_nonzero"],
        )

        # Save preprocessed data
        save_path = join(target_preprocessed, f"{prefix}_{subject_id}")
        np.save(save_path + ".npy", preprocessed_images)

        # Save label for preprocessed data
        with open(join(target_preprocessed, f"{prefix}_{subject_id}.txt"), "w") as f:
            f.write(str(age))

        return f"Processed {subject}"

    except Exception as e:
        return f"Error processing {subject}: {str(e)}"


def convert_and_preprocess_task3(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess all subjects for Task 3 in parallel.

    Args:
        source_path: Path to the source data directory
        output_path: Path where preprocessed data will be saved (optional)
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    # Get configuration from task3_config
    pp_config = task3_config
    task_name = pp_config["task_name"]
    prefix = "FOMO3"

    # Input data paths
    labels_dir = join(source_path, "labels")
    images_dir = join(source_path, "preprocessed")

    # Output path for preprocessed data
    target_preprocessed = join(output_path, task_name)

    # Create directory
    ensure_dir_exists(target_preprocessed)

    # Process all subjects
    subject_folders = [f for f in os.listdir(images_dir) if f.startswith("sub_")]

    # Create task information for each subject
    tasks = [
        (subject, source_path, labels_dir, pp_config, target_preprocessed, prefix)
        for subject in subject_folders
    ]

    # Process all subjects in parallel
    parallel_process(
        process_subject, tasks, num_workers, desc="Processing subjects for Task 3"
    )

    print(f"Task 3 preprocessing completed. Data saved to {target_preprocessed}")
