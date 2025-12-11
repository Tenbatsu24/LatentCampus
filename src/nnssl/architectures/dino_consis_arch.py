import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
from dynamic_network_architectures.architectures.unet import (
    PlainConvUNet,
    ResidualEncoderUNet,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=2**16,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, **kwargs):
        x = self.mlp(x)
        eps = torch.finfo(x.dtype).eps
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


def _build_mlp(
    nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True
):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
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


class MaskingPlainDecoder(UNetDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dino_heads = nn.ModuleList(
            [
                DINOHead(
                    self.encoder.output_channels[-(len(self.encoder.output_channels))],
                    self.num_classes,
                )
            ]
        )

    def forward(self, skips, mask=None):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :param mask:
        :return:
        """
        if mask is None:
            return super().forward(skips)

        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                raise NotImplementedError(
                    f"MaskingPlainDecoder does not handle {self.deep_supervision=}"
                )
            elif s == (len(self.stages) - 1):
                feat_last = x.permute(0, 2, 3, 4, 1)
                selected_locs = feat_last[
                    mask.unsqueeze(-1).expand_as(feat_last)
                ].reshape(x.shape[0], -1, x.shape[1])
                seg_outputs.append(self.dino_heads[-1](selected_locs))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


class DINOConsisPlainMAE(PlainConvUNet):

    def __init__(
        self,
        input_channels=1,
        n_stages=7,
        features_per_stage=(32, 64, 128, 256, 320, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=None,
        strides=(
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
        num_classes=2**11,
        n_conv_per_stage=(2, 2, 2, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2, 2),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=False,
        **kwargs,
    ):
        if kernel_sizes is None:
            kernel_sizes = [[1, 3, 3], *[[3, 3, 3] for _ in range(n_stages - 1)]]
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
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )

        self.i_adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        proj_in_dim = sum(features_per_stage)

        self.dino_head = DINOHead(
            in_dim=proj_in_dim,
            out_dim=num_classes,
        )
        self.dino_head.init_weights()

        self.decoder = MaskingPlainDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            nonlin_first=False,
        )

    def forward(self, x, mask=None):
        b = x.shape[0]
        skips = self.encoder(x)
        print([s.shape for s in skips])
        voxel_cls = self.decoder(skips, mask)

        image_latent = torch.concat(
            [self.i_adaptive_pool(s) for s in skips], dim=1
        ).reshape(b, -1)

        global_cls = self.dino_head(image_latent)

        return {
            "latent": image_latent,
            "proj_pred": global_cls,
            "patch_latent": voxel_cls,
        }


class DINOConsisResMAE(ResidualEncoderUNet):

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

        self.i_adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if only_last_stage_as_latent:
            proj_in_dim = features_per_stage[-1]
        else:
            proj_in_dim = sum(features_per_stage)
        self.only_last_stage_as_latent = only_last_stage_as_latent

        self.dino_head = DINOHead(
            in_dim=proj_in_dim,
        )

        self.dino_head.init_weights()

        self.decoder = MaskingPlainDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision
        )

    def forward(self, x):
        b = x.shape[0]
        skips = self.encoder(x)
        print([s.shape for s in skips])
        voxel_cls = self.decoder(skips)

        image_latent = torch.concat(
            [self.i_adaptive_pool(s) for s in skips], dim=1
        ).reshape(b, -1)

        global_cls = self.dino_head(image_latent)

        return {
            "latent": image_latent,
            "proj_pred": global_cls,
            "patch_latent": voxel_cls,
        }


if __name__ == "__main__":
    import os
    import gc
    import psutil

    import thop

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def measure_memory(model, *input_tensors):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(*input_tensors)
        mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # in MB
        mem_peak = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
        print(f"Current allocated memory: {mem_allocated:.2f} MB")
        print(f"Peak memory usage: {mem_peak:.2f} MB")

    def measure_memory_cpu(model, *input_tensors):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)  # in MB
        with torch.no_grad():
            _ = model(*input_tensors)
        mem_after = process.memory_info().rss / (1024**2)  # in MB
        print(f"Memory before: {mem_before:.2f} MB")
        print(f"Memory after: {mem_after:.2f} MB")
        print(f"Memory used by forward pass: {mem_after - mem_before:.2f} MB")

    input_shape = (20, 128, 128)
    input_tensor = torch.randn(1, 1, *input_shape, device=_device)
    mask = torch.randint(0, 2, (1, *input_shape), device=_device).to(torch.bool)

    model = DINOConsisPlainMAE(num_classes=2**13)
    model.train()
    print(model)
    model = model.to(_device)

    # # make the decoder an identity function
    # model.decoder = nn.Identity()
    # model.train(False)
    if _device == "cuda":
        measure_memory(model, input_tensor, mask)
    else:
        measure_memory_cpu(model, input_tensor, mask)

    flops, params = thop.profile(model, inputs=(input_tensor, mask), verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")
    del model
    gc.collect()
    torch.cuda.empty_cache()
