from typing import Tuple

import torch.nn as nn

from einops import rearrange
from torch.nn.utils import weight_norm
from torch.nn.init import trunc_normal_
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
            "latent": latent,
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
        **kwargs,
    ):
        super().__init__(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_embed_size=patch_embed_size,
            output_channels=output_channels,
            **kwargs,
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

        # Make projection
        projection = self.projector(encoded)["proj"]

        # Restore full sequence with mask tokens
        num_patches = w * h * d
        if self.use_decoder:
            restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)

            # Decode with restored sequence and rope embeddings
            decoded, _ = self.decoder(restored_x)
        else:
            decoded = encoded

        # Project back to output shape
        decoded = rearrange(decoded, "b (h w d) c -> b c w h d", h=w, w=h, d=d)
        decoded = self.up_projection(decoded)

        return {
            "patch_latent": encoded,
            "proj": projection,
            "recon": decoded,
            "keep_indices": keep_indices,
        }


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # Toy example for testing
    input_shape = (64, 64, 64)
    patch_embed_size = (8, 8, 8)
    model = ConsisEvaMAE(
        input_channels=3,
        embed_dim=192,
        patch_embed_size=patch_embed_size,
        output_channels=3,
        input_shape=input_shape,
        decoder_eva_depth=6,
        decoder_eva_numheads=8,
        patch_drop_rate=0.7,
    )

    # Random input tensor
    x = torch.rand((2, 3, *input_shape))  # Batch size 2

    # Forward pass
    output = model(x)
    print("Input shape:", x.shape)
    print(
        f"Output shape: {output['recon'].shape}, "
        f"Keep indices shape: {output['keep_indices'].shape}, "
        f"Latent shape: {output['patch_latent'].shape}",
        f"Projection shape: {output['proj'].shape}",
    )

    model = ConsisMAE(
        input_channels=1,
        num_classes=1,
        deep_supervision=False,
        only_last_stage_as_latent=True,
    )
    x = torch.rand((2, 1, *input_shape))  # Batch size 2
    output = model(x)
    print("Input shape:", x.shape)
    print(
        f"Output shape: {output['recon'].shape}, "
        f"Latent shape: {output['latent'].shape}",
        f"Projection shape: {output['proj'].shape}",
    )  # Latent is a list of tensors
