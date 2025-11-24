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
    extracted_fomo_task_2_dir = str(Path(".").resolve())
    nnunet_dataset_name = "FOMO60k2"
    nnunet_dataset_id = 602
    dataset_name = f"Dataset{nnunet_dataset_id:03d}_{nnunet_dataset_name}"
    dataset_dir = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(dataset_dir)

    dataset = {}
    casenames = list(
        map(
            lambda _pth: str(_pth.name),
            Path(extracted_fomo_task_2_dir, "preprocessed").glob("sub_*"),
        )
    )
    for c in casenames:
        last_modality_path = Path(
            extracted_fomo_task_2_dir, "preprocessed", c, "ses_1", "swi.nii.gz"
        )

        if not last_modality_path.exists():
            # it is either a t2s or a swi file, so we check for t2s
            last_modality_path = Path(
                extracted_fomo_task_2_dir, "preprocessed", c, "ses_1", "t2s.nii.gz"
            )

            if not last_modality_path.exists():
                raise FileNotFoundError(
                    f"Neither SWI nor T2S modality found for {c} in {extracted_fomo_task_2_dir}"
                )
            else:
                last_modality = "t2s"

        else:
            last_modality = "swi"

        dataset[c] = {
            "label": join(
                extracted_fomo_task_2_dir, "labels", c, "ses_1", "seg.nii.gz"
            ),
            "images": [
                join(
                    extracted_fomo_task_2_dir,
                    "preprocessed",
                    c,
                    "ses_1",
                    "dwi_b1000.nii.gz",
                ),
                join(
                    extracted_fomo_task_2_dir,
                    "preprocessed",
                    c,
                    "ses_1",
                    "flair.nii.gz",
                ),
                join(
                    extracted_fomo_task_2_dir,
                    "preprocessed",
                    c,
                    "ses_1",
                    f"{last_modality}.nii.gz",
                ),
            ],
        }

    labels = {
        "background": 0,
        "meningioma": 1,
    }

    for c in casenames:
        label_path = dataset[c]["label"]
        images = dataset[c]["images"]
        sizes = []
        for image_path in images:
            image = sitk.ReadImage(image_path)
            sizes.append(sitk.GetArrayFromImage(image).shape)

            # print the size of the image
            print(f"Size of {image_path}: {sizes[-1]}")

        # print the sizes of all the images and the label
        label_image = sitk.ReadImage(label_path)
        print(
            f"Size of label {label_path}: {sitk.GetArrayFromImage(label_image).shape}"
        )

        sizes.append(sitk.GetArrayFromImage(label_image).shape)

        # find the number of unqiue classes in the label
        unique_classes = set(sitk.GetArrayFromImage(label_image).flatten())
        print(f"Unique classes in label {label_path}: {unique_classes}")

        # check if all sizes are equal
        if not all(size == sizes[0] for size in sizes):
            print("====" * 20)
            print(f"Sizes for {c} are not equal:")
            for i, size in enumerate(sizes):
                print(f"Image {i}: {size}")
            print("====" * 20)

    generate_dataset_json(
        dataset_dir,
        {0: "dwi", 1: "flair", 2: "swi"},
        labels,
        num_training_cases=len(dataset),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=dataset_name,
        reference="https://www.synapse.org/Synapse:syn64895667/wiki/",
        license="see https://www.synapse.org/Synapse:syn64895667/wiki/",
        dataset=dataset,
        description="This dataset does not copy the data into nnunet format and just links to existing data. "
        "The dataset can only be used from one machine because the paths in the dataset.json are hard coded",
    )
