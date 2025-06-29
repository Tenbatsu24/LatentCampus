import torch.nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.architectures.primus import (
    PrimusS,
    PrimusB,
    PrimusM,
    PrimusL,
)
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)

from nnssl.architectures.architecture_registry import (
    SUPPORTED_ARCHITECTURES,
    get_res_enc_l,
    get_noskip_res_enc_l,
)
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan


def get_network_by_name(
    configuration_plan: ConfigurationPlan,
    architecture_name: SUPPORTED_ARCHITECTURES,
    num_input_channels: int,
    num_output_channels: int,
    encoder_only: bool = False,
    deep_supervision: bool = False,
    arch_kwargs: dict | None = None,
) -> AbstractDynamicNetworkArchitectures:
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    if architecture_name == "ResEncL":
        model = get_res_enc_l(num_input_channels, num_output_channels, deep_supervision)
    elif architecture_name == "NoSkipResEncL":
        model = get_noskip_res_enc_l(num_input_channels, num_output_channels)
    elif architecture_name in ["PrimusS", "PrimusB", "PrimusM", "PrimusL"]:
        if architecture_name == "PrimusS":
            model = PrimusS(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        elif architecture_name == "PrimusB":
            model = PrimusB(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        elif architecture_name == "PrimusM":
            model = PrimusM(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        elif architecture_name == "PrimusL":
            model = PrimusL(
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_shape=configuration_plan.patch_size,
                patch_embed_size=(8, 8, 8),
            )
        else:
            raise ValueError(f"Architecture {architecture_name} is not supported.")
    else:
        raise ValueError(f"Architecture {architecture_name} is not supported.")

    if encoder_only:
        if architecture_name in ["ResEncL", "NoSkipResEncL"]:
            model: ResidualEncoderUNet
            try:
                model.decoder = torch.nn.Identity()
                # key_to_encoder = model.key_to_encoder.replace("encoder.", "")
                # keys_to_in_proj = [k.replace("encoder.", "") for k in model.keys_to_in_proj]
                # model = model.encoder
                # model.key_to_encoder = key_to_encoder
                # model.keys_to_in_proj = keys_to_in_proj
            except AttributeError:
                raise RuntimeError(
                    "Trying to get the 'encoder' of the network failed. Cannot return encoder only."
                )
        elif architecture_name in ["PrimusS", "PrimusB", "PrimusM", "PrimusL"]:
            raise NotImplementedError(
                "Cannot return encoder only for Primus architectures."
            )
    return model


if __name__ == "__main__":
    import thop
    import torch
    import torch.nn as nn

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def measure_memory(model, input_tensor):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_tensor)
        mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # in MB
        mem_peak = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
        print(f"Current allocated memory: {mem_allocated:.2f} MB")
        print(f"Peak memory usage: {mem_peak:.2f} MB")

    # Toy example for testing
    input_shape = (64, 64, 64)

    model = get_res_enc_l(1, 1, deep_supervision=False).to(_device)

    x = torch.rand((2, 1, *input_shape), device=_device)  # Batch size 2
    # output = model(x)
    # print("Input shape:", x.shape)
    # print(
    #     f"Output shape: {output['recon'].shape}, "
    #     f"Latent shape: {output['latent'].shape}",
    #     f"Projection shape: {output['proj'].shape}",
    # )  # Latent is a list of tensors
    if _device == "cuda":
        measure_memory(model, x)
    macs, params = thop.profile(
        model,
        inputs=(x,),
    )
    print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
