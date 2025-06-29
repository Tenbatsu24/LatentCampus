from typing import Tuple

import torch
import torch.nn as nn

from einops import rearrange
from torch.nn.utils import weight_norm
from torch.nn.init import trunc_normal_
from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnssl.architectures.evaMAE_module import EvaMAE


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        use_bn=False,
        n_layers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        l2_normalize=True,
    ):
        super().__init__()
        n_layers = max(n_layers, 1)

        self.out_dim = out_dim
        self.mlp = _build_mlp(
            n_layers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.l2_normalize = l2_normalize
        self.apply(self._init_weights)

        if out_dim is None:
            self.last_layer = nn.Identity()
        else:
            self.last_layer = weight_norm(
                nn.Linear(bottleneck_dim, out_dim, bias=False)
            )
            self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)

        if self.l2_normalize:
            # Use a small epsilon to avoid division by zero
            # This is especially important for float16 inputs
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)

        if self.out_dim is not None:
            classifier = self.last_layer(x)
        else:
            classifier = None

        return {
            "logits": classifier,
            "proj": x,
        }


def _build_mlp(
    nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True
):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers: list["nn.Module"] = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


class ConsisMAE(ResidualEncoderUNet):

    def __init__(
        self,
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=None,
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        num_classes=1,
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=False,
        only_last_stage_as_latent=False,
    ):
        if kernel_sizes is None:
            kernel_sizes = [[3, 3, 3] for _ in range(n_stages)]
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}

        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if only_last_stage_as_latent:
            proj_in_dim = features_per_stage[-1]
        else:
            proj_in_dim = sum(features_per_stage)
        self.only_last_stage_as_latent = only_last_stage_as_latent
        self.projector = ProjectionHead(
            in_dim=proj_in_dim,
            out_dim=None,
        )

    def forward(self, x):
        skips = self.encoder(x)
        decoded = self.decoder(skips)
        if self.only_last_stage_as_latent:
            skips = [skips[-1]]
        latent = torch.concat(
            [self.adaptive_pool(s) for s in reversed(skips)], dim=1
        ).reshape(x.shape[0], -1)
        projection = self.projector(latent)
        return {
            "proj": projection["proj"],
            "recon": decoded,
        }


class ConsisEvaMAE(EvaMAE):

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        output_channels: int,
        input_shape: Tuple[int, int, int] = None,
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            output_channels=output_channels,
            input_shape=input_shape,
            **kwargs,
        )

        if not self.use_decoder:
            raise ValueError("ConsisEvaMAE requires a decoder to be used.")

        self.feature_decoder = Eva(
            embed_dim=embed_dim,
            depth=1,  # eva_depth,
            num_heads=16,  # eva_numheads,
            ref_feat_shape=tuple(
                [i // ds for i, ds in zip(input_shape, patch_embed_size)]
            ),
            num_reg_tokens=kwargs.get("num_register_tokens", 0),
            use_rot_pos_emb=kwargs.get("use_rot_pos_emb", True),
            use_abs_pos_emb=kwargs.get("use_abs_pos_emb", True),
            mlp_ratio=kwargs.get("mlp_ratio", 4 * 2 / 3),
            drop_path_rate=kwargs.get("drop_path_rate", 0),
            patch_drop_rate=0,  # No drop in the decoder
            proj_drop_rate=kwargs.get("proj_drop_rate", 0.0),
            attn_drop_rate=kwargs.get("attn_drop_rate", 0.0),
            init_values=kwargs.get("init_values", 0.1),
            scale_attn_inner=kwargs.get("scale_attn_inner", False),
        )
        self.projector = ProjectionHead(
            in_dim=embed_dim,
            out_dim=None,
        )

    def forward(self, x):
        # Encode patches
        x = self.down_projection(x)
        b, c, w, h, d = x.shape
        x = rearrange(x, "b c w h d -> b (h w d) c")

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.eva(x)
        num_patches = w * h * d

        if self.training:
            # Restore full sequence with mask tokens
            restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)
            features_decoded, _ = self.feature_decoder(restored_x)
            projection = self.projector(features_decoded)["proj"]
        else:
            restored_x = encoded
            projection = self.projector(encoded)["proj"]

        # Decode with restored sequence and rope embeddings
        decoded, _ = self.decoder(restored_x)

        # Project back to output shape
        decoded = rearrange(decoded, "b (h w d) c -> b c w h d", h=w, w=h, d=d)
        decoded = self.up_projection(decoded)
        # Reshape restored sequence to match original patch shape
        projection = rearrange(projection, "b (h w d) c -> b c w h d", h=w, w=h, d=d)

        return {
            "proj": projection,
            "recon": decoded,
            "keep_indices": keep_indices,
        }


if __name__ == "__main__":
    import thop

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

    model = ConsisMAE(
        input_channels=1,
        num_classes=1,
        deep_supervision=False,
        only_last_stage_as_latent=False,
    ).to(_device)
    x = torch.rand((2, 1, *input_shape), device=_device)  # Batch size 2
    output = model(x)
    print("Input shape:", x.shape)
    print(
        f"Output shape: {output['recon'].shape}, "
        f"Projection shape: {output['proj'].shape}",
    )  # Latent is a list of tensors
    if _device == "cuda":
        measure_memory(model, x)
    macs, params = thop.profile(
        model,
        inputs=(x,),
    )
    print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

    patch_embed_size = (8, 8, 8)
    model = ConsisEvaMAE(
        input_channels=1,
        embed_dim=192,
        patch_embed_size=patch_embed_size,
        output_channels=1,
        input_shape=input_shape,
        decoder_eva_depth=6,
        decoder_eva_numheads=8,
        patch_drop_rate=0.7,
    ).to(_device)

    # Random input tensor
    x = torch.rand((2, 1, *input_shape), device=_device)  # Batch size 2

    # Forward pass
    # measure the memory

    output = model(x)
    print("Input shape:", x.shape)
    print(
        f"Output shape: {output['recon'].shape}, "
        f"Keep indices shape: {output['keep_indices'].shape}, "
        f"Latent shape: {output['proj'].shape}",
    )
    if _device == "cuda":
        measure_memory(model, x)
    macs, params = thop.profile(
        model,
        inputs=(x,),
    )
    print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
