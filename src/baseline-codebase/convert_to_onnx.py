import torch

from pathlib import Path
from src.models.networks import resenc_l

def get_in_out_mode(task):
    if task == 1:
        return 4, 2, "classification"
    elif task == 3:
        return 3, 2, "regression"
    else:
        raise ValueError("Unsupported task number")


def initialise_model(path_to_model_weights, task):
    input_channels, num_classes, mode = get_in_out_mode(task)

    model = resenc_l(
        mode=mode,
        input_channels=input_channels,
        num_classes=num_classes,
        output_channels=num_classes,
        deep_supervision=False
    )

    # remove 'model.' prefix if it exists in the state_dict keys
    new_state_dict = dict()

    state_dict = torch.load(path_to_model_weights, map_location="cpu")
    for k, v in state_dict.items():
        new_key = k[6:]
        new_state_dict[new_key] = v

    # check if zip(new_state_dict.keys(), model.state_dict().keys()) all match
    for k1, k2 in zip(new_state_dict.keys(), model.state_dict().keys()):
        if k1 != k2:
            print(f"Key mismatch: {k1} != {k2}")

    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    model.requires_grad_(False)
    # make sure all norm layers are in eval mode and not tracking running stats and not training
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.eval()
        if isinstance(m, torch.nn.modules.instancenorm._InstanceNorm):
            m.track_running_stats = False
            m.eval()

    return model, input_channels


def convert_to_onnx(path_to_model_weights, task):
    model, input_channels = initialise_model(path_to_model_weights, task)

    # example path = /.../Task00{task}_FOMO{task}/resenc_l/version_{version}/checkpoints/last.ckpt
    pathlib_path = Path(path_to_model_weights)
    # save it to /.../Task00{task}_FOMO{task}/onnx_models/LatentCampus_Task00{task}_v{version}.onnx
    path_to_dir = pathlib_path.parent.parent.parent / "onnx_models"
    path_to_dir.mkdir(parents=True, exist_ok=True)
    path_to_save = path_to_dir / f"LatentCampus_Task00{task}_v{pathlib_path.parent.parent.name.split('_')[-1]}.onnx"

    # Create example inputs for exporting the model. The inputs should be a tuple of tensors.
    example_inputs = (torch.randn(1, input_channels, 96, 96, 96),)
    onnx_program = torch.onnx.export(model, example_inputs, path_to_save)

    print(onnx_program)
    print(f"ONNX model saved to {path_to_save}")

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--task", type=int, choices=[1, 3], required=True, help="Task number (1 for classification, 3 for regression)")
    args = parser.parse_args()

    onnx_model = convert_to_onnx(args.model_weights, args.task)
