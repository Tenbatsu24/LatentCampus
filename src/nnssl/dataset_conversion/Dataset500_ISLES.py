from pathlib import Path

import SimpleITK as sitk

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subdirs,
    subfiles,
    maybe_mkdir_p,
)
from nnunetv2.paths import nnUNet_raw


if __name__ == "__main__":
    """
    this dataset does not copy the data into nnunet format and just links to existing data. The dataset can only be
    used from one machine because the paths in the dataset.json are hard coded
    """
    extracted_isles_data_dir = "/deepstore/datasets/mia/HealthyAI/ISLES-2022"
    nnunet_dataset_name = "ISLES-2022"
    nnunet_dataset_id = 500
    dataset_name = f"Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}"
    dataset_dir = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(dataset_dir)

    dataset = {}
    casenames = list(
        map(
            lambda _pth: str(_pth.name),
            Path(extracted_isles_data_dir).glob("*sub-strokecase*"),
        )
    )
    label_paths = [
        join(
            extracted_isles_data_dir,
            "derivatives",
            c,
            "ses-0001",
            f"{c}_ses-0001_msk.nii.gz",
        )
        for c in casenames
    ]
    for c, label_path in zip(casenames, label_paths):
        dataset[c] = {
            "label": label_path,
            "images": [
                # join(extracted_isles_data_dir, c, 'ses-0001', 'anat', f"{c}_ses-0001_FLAIR.nii.gz"),
                join(
                    extracted_isles_data_dir,
                    c,
                    "ses-0001",
                    "dwi",
                    f"{c}_ses-0001_dwi.nii.gz",
                ),
                join(
                    extracted_isles_data_dir,
                    c,
                    "ses-0001",
                    "dwi",
                    f"{c}_ses-0001_adc.nii.gz",
                ),
            ],
        }

    labels = {
        "background": 0,
        "stroke": 1,
    }

    # resize all dwi to target spacing [1, 1, 1] and then resize all adc to dwi spacing
    # target_spacing = np.array([1, 1, 1])
    for c in casenames:
        label_path = dataset[c]["label"]
        dwi_path = dataset[c]["images"][0]
        adc_path = dataset[c]["images"][1]

        # print the sizes of all the images and the label
        dwi_image = sitk.ReadImage(dwi_path)
        adc_image = sitk.ReadImage(adc_path)
        label_image = sitk.ReadImage(label_path)

        sizes = [dwi_image.GetSize(), adc_image.GetSize(), label_image.GetSize()]

        # check if all sizes are equal
        if not all(size == sizes[0] for size in sizes):
            print("====" * 20)
            print(
                f"Warning: Sizes are not equal for case {c}. DWI size: {dwi_image.GetSize()}, ADC size: {adc_image.GetSize()}, Label size: {label_image.GetSize()}"
            )
            print("====" * 20)

    generate_dataset_json(
        dataset_dir,
        {0: "DWI", 1: "ADC"},
        labels,
        num_training_cases=len(dataset),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=dataset_name,
        reference="https://arxiv.org/abs/2206.06694",
        license="see https://arxiv.org/abs/2206.06694",
        dataset=dataset,
        description="This dataset does not copy the data into nnunet format and just links to existing data. "
        "The dataset can only be used from one machine because the paths in the dataset.json are hard coded",
    )
