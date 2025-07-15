from typing import Tuple

import torch
import torch.nn as nn

from einops import rearrange
from torch.nn.init import trunc_normal_
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnssl.architectures.evaMAE_module import EvaMAE


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
        use_projector=False,
        **kwargs,
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

        self.v_adaptive_pool = nn.AdaptiveAvgPool3d((20, 20, 20))
        self.i_adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.use_projector = use_projector
        if only_last_stage_as_latent:
            proj_in_dim = features_per_stage[-1]
        else:
            proj_in_dim = sum(features_per_stage)
        self.only_last_stage_as_latent = only_last_stage_as_latent

        if self.use_projector:
            self.projector = nn.Sequential(
                nn.Linear(proj_in_dim, 2048),  # this is technically a linear layer
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048, affine=False, track_running_stats=False),
            )  # output layer

            self.predictor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512, affine=False, track_running_stats=False),
                nn.SiLU(),
                nn.Linear(512, 2048),
            )

            # initialize the projector weights
            for m in self.projector.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips = self.encoder(x)
        decoded = self.decoder(skips)

        if self.only_last_stage_as_latent:
            skips = [skips[-1]]
        latent = torch.concat(
            [self.v_adaptive_pool(s) for s in skips], dim=1
        )

        if self.use_projector:
            b = latent.shape[0]
            latent = rearrange(latent, "b c w h d -> (b w h d) c")
            latent = self.projector(latent)

            if self.training:
                latent = self.predictor(latent)

            latent = rearrange(latent, "(b w h d) c -> b c w h d", b=b, w=20, h=20, d=20)

        image_latent = torch.concat(
            [self.i_adaptive_pool(s) for s in skips], dim=1
        ).reshape(x.shape[0], -1)

        if self.use_projector:
            image_latent = self.projector(image_latent)
            if self.training:
                image_latent = self.predictor(image_latent)

        return {
            "proj": latent,
            "recon": decoded,
            "image_latent": image_latent,
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

        # self.feature_decoder = Eva(
        #     embed_dim=embed_dim,
        #     depth=1,  # eva_depth,
        #     num_heads=16,  # eva_numheads,
        #     ref_feat_shape=tuple(
        #         [i // ds for i, ds in zip(input_shape, patch_embed_size)]
        #     ),
        #     num_reg_tokens=kwargs.get("num_register_tokens", 0),
        #     use_rot_pos_emb=kwargs.get("use_rot_pos_emb", True),
        #     use_abs_pos_emb=kwargs.get("use_abs_pos_emb", True),
        #     mlp_ratio=kwargs.get("mlp_ratio", 4 * 2 / 3),
        #     drop_path_rate=kwargs.get("drop_path_rate", 0),
        #     patch_drop_rate=0,  # No drop in the decoder
        #     proj_drop_rate=kwargs.get("proj_drop_rate", 0.0),
        #     attn_drop_rate=kwargs.get("attn_drop_rate", 0.0),
        #     init_values=kwargs.get("init_values", 0.1),
        #     scale_attn_inner=kwargs.get("scale_attn_inner", False),
        # )

        self.use_projector = True

        self.projector = nn.Sequential(
            nn.Conv1d(embed_dim, 2048, kernel_size=1, bias=False),  # this is technically a linear layer
            nn.InstanceNorm1d(2048, affine=False, track_running_stats=False),
            nn.SiLU(),
            nn.Conv1d(2048, 2048, kernel_size=1, bias=False),
            nn.InstanceNorm1d(2048, affine=False, track_running_stats=False),
            nn.SiLU(),
            nn.Conv1d(2048, 2048, kernel_size=1, bias=False),
            nn.InstanceNorm1d(2048, affine=False, track_running_stats=False),
        )  # output layer

        self.predictor = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=1, bias=False),
            nn.InstanceNorm1d(512, affine=False, track_running_stats=False),
            nn.SiLU(),
            nn.Conv1d(512, 2048, kernel_size=1, bias=False),
        )

        # initialize the projector weights
        for m in self.projector.modules():
            if isinstance(m, nn.Conv1d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.i_adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # Encode patches
        x = self.down_projection(x)
        b, c, w, h, d = x.shape
        x = rearrange(x, "b c w h d -> b (h w d) c")

        # Encode using EVA (internally applies masking with patch_drop_rate)
        encoded, keep_indices = self.eva(x)
        # print(f"Encoded shape: {encoded.shape}, Keep indices shape: {keep_indices.shape if keep_indices is not None else 'None'}")
        num_patches = w * h * d

        if keep_indices is None or not self.training:
            restored_x = encoded
        else:
            # Restore full sequence with mask tokens
            restored_x = self.restore_full_sequence(encoded, keep_indices, num_patches)

        # Decode with restored sequence and rope embeddings
        decoded, _ = self.decoder(restored_x)

        if self.use_projector:
            decoded = rearrange(decoded, "b s c -> b c s")
            projected = self.projector(decoded)

            if self.training:
                projected = self.predictor(projected)

            projected = rearrange(projected, "b c (h w d) -> b c w h d", b=b, w=w, h=h, d=d)

            image_latent = self.i_adaptive_pool(projected).reshape(b, -1)
        else:
            projected = None
            image_latent = None

        # Project back to output shape
        decoded = rearrange(decoded, "b c (h w d) -> b c w h d", h=w, w=h, d=d)
        decoded = self.up_projection(decoded)

        return {
            "proj": projected,
            "image_latent": image_latent,
            "recon": decoded,
            "keep_indices": keep_indices,
        }


if __name__ == "__main__":
    # import os
    # import psutil
    #
    # import thop
    #
    # from nnssl.architectures.architecture_registry import get_res_enc_l

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # def measure_memory(model, input_tensor):
    #     torch.cuda.reset_peak_memory_stats()
    #     with torch.no_grad():
    #         _ = model(input_tensor)
    #     mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # in MB
    #     mem_peak = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    #     print(f"Current allocated memory: {mem_allocated:.2f} MB")
    #     print(f"Peak memory usage: {mem_peak:.2f} MB")
    #
    #
    # def measure_memory_cpu(model, input_tensor):
    #     process = psutil.Process(os.getpid())
    #     mem_before = process.memory_info().rss / (1024 ** 2)  # in MB
    #     with torch.no_grad():
    #         _ = model(input_tensor)
    #     mem_after = process.memory_info().rss / (1024 ** 2)  # in MB
    #     print(f"Memory before: {mem_before:.2f} MB")
    #     print(f"Memory after: {mem_after:.2f} MB")
    #     print(f"Memory used by forward pass: {mem_after - mem_before:.2f} MB")
    #
    # Toy example for testing
    input_shape = (64, 64, 64)

    model = ConsisMAE(
        input_channels=1,
        num_classes=1,
        deep_supervision=False,
        only_last_stage_as_latent=False,
        use_projector=True
    ).to(_device)
    model = model.train(True)
    x = torch.rand((2, 1, *input_shape), device=_device)  # Batch size 2
    output = model(x)
    print(
        f"Input shape: {x.shape}, "
        f"Output shape: {output['recon'].shape}, "
        f"Latent shape: {output['proj'].shape}, "
        f"Image latent shape: {output['image_latent'].shape if output['image_latent'] is not None else 'None'}",
    )
    # del output
    # if _device == "cuda":
    #     measure_memory(model, x)
    # else:
    #     measure_memory_cpu(model, x)
    # macs, params = thop.profile(
    #     model,
    #     inputs=(x,),
    # )
    # print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    #
    # model = get_res_enc_l(1, 1, deep_supervision=False).to(_device)
    # if _device == "cuda":
    #     measure_memory(model, x)
    # else:
    #     measure_memory_cpu(model, x)
    # macs, params = thop.profile(
    #     model,
    #     inputs=(x,),
    # )
    # print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    #
    # patch_embed_size = (8, 8, 8)
    # model = ConsisEvaMAE(
    #     input_channels=1,
    #     embed_dim=192,
    #     patch_embed_size=patch_embed_size,
    #     output_channels=1,
    #     input_shape=input_shape,
    #     decoder_eva_depth=6,
    #     decoder_eva_numheads=8,
    #     patch_drop_rate=0.7,
    # ).to(_device)
    #
    # # Random input tensor
    # x = torch.rand((2, 1, *input_shape), device=_device)  # Batch size 2
    #
    # # Forward pass
    # # measure the memory
    #
    # output = model(x)
    # print("Input shape:", x.shape)
    # print(
    #     f"Output shape: {output['recon'].shape}, "
    #     f"Keep indices shape: {output['keep_indices'].shape}, "
    #     f"Latent shape: {output['proj'].shape}",
    # )
    # if _device == "cuda":
    #     measure_memory(model, x)
    # else:
    #     measure_memory_cpu(model, x)
    # macs, params = thop.profile(
    #     model,
    #     inputs=(x,),
    # )
    # print(f"MACs: {macs / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

    patch_embed_size = (8, 8, 8)
    model = ConsisEvaMAE(
        input_channels=1,
        embed_dim=192,
        patch_embed_size=patch_embed_size,
        output_channels=1,
        input_shape=input_shape,
        decoder_eva_depth=6,
        decoder_eva_numheads=8,
        patch_drop_rate=0.0,
    ).to(_device)
    model = model.train(True)

    # Random input tensor
    x = torch.rand((2, 1, *input_shape), device=_device)  # Batch size 2
    # Forward pass
    output = model(x)
    print("Input shape:", x.shape)
    print(
        f"Output shape: {output['recon'].shape}, "
        f"Keep indices shape: {output['keep_indices'].shape if output['keep_indices'] is not None else 'None'}, "
        f"Latent shape: {output['proj'].shape}",
        f"Image latent shape: {output['image_latent'].shape if output['image_latent'] is not None else 'None'}",
    )

    model = model.train(False)
    output = model(x)
    print("Input shape:", x.shape)
    print(
        f"Output shape: {output['recon'].shape if output['recon'] is not None else 'None'}, "
        f"Keep indices shape: {output['keep_indices'].shape if output['keep_indices'] is not None else 'None'}, "
        f"Latent shape: {output['proj'].shape if output['proj'] is not None else 'None'}",
        f"Image latent shape: {output['image_latent'].shape if output['image_latent'] is not None else 'None'}",
    )
