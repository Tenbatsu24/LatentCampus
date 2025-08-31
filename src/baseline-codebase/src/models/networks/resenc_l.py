import torch
import torch.nn as nn

from yucca.modules.networks.networks.YuccaNet import YuccaNet
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class GlobalPoolAndConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # x is a list of features from all the stages
        pooled_features = [self.global_pool(feature) for feature in x]
        # Concatenate the pooled features along the channel dimension
        x = torch.cat(pooled_features, dim=1)
        # Flatten the concatenated features
        x = torch.flatten(x, start_dim=1)
        return x


class MinusOneOneToZeroHundred(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x) * 100.0
        return x


class ClsRegHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_layers=3, hidden_dim=512, dropout=0.5, **kwargs):
        super().__init__()
        self.num_layers = num_layers

        if num_layers == 1:
            self.fc = nn.Linear(in_channels, num_classes)
            nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias, 0)
        else:
            layers = []
            for i in range(num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_channels, hidden_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            self.fc = nn.Sequential(*layers, nn.Linear(hidden_dim, num_classes, bias=False), MinusOneOneToZeroHundred())

            # initialization of the fully connected layer
            for m in self.fc:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        return x


class ResEncL(YuccaNet):
    """
    Wrapper for the ResEnc-L architecture to be compatible with YuccaNet and the repository's finetuning code.
    """

    def __init__(
        self,
        mode: str = "segmentation",
        input_channels: int = 1,
        num_classes: int = 1,
        output_channels: int = 1,
        deep_supervision: bool = False,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.num_features_per_stage = [32, 64, 128, 256, 320, 320]
        original_model = ResidualEncoderUNet(
            input_channels=input_channels,
            n_stages=6,
            features_per_stage=self.num_features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[3, 3, 3] for _ in range(6)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=output_channels,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )
        self.encoder = original_model.encoder

        if mode == "segmentation":
            self.decoder = original_model.decoder
        elif mode in ["classification", "regression"]:
            self.decoder = nn.Sequential(
                GlobalPoolAndConcat(),
                ClsRegHead(
                    in_channels=sum(self.num_features_per_stage),
                    num_classes=num_classes,
                    num_layers=3 if mode == "regression" else 1,
                )
            )
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Supported modes are 'segmentation', 'classification', and 'regression'."
            )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def load_state_dict(self, target_state_dict, *args, **kwargs):
        # use the Module's load_state_dict method not the parent class YuccaNet's method
        nn.Module.load_state_dict(self, target_state_dict, *args, **kwargs)


def resenc_l(
        mode: str = "segmentation",
        input_channels: int = 1,
        num_classes: int = 1,
        output_channels: int = 1,
        deep_supervision: bool = False,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        **kwargs
):
    """
    Factory function for ResEnc-L, compatible with finetuning code.
    """
    return ResEncL(
        mode=mode,
        input_channels=input_channels,
        num_classes=num_classes,
        output_channels=output_channels,
        deep_supervision=deep_supervision,
        conv_op=conv_op,
        norm_op=norm_op,
        **kwargs
    )


if __name__ == '__main__':
    # from pathlib import Path
    #
    # _example_state_dict_path = Path("/mnt/c/Users/puruv/Projects/LatentCampus/models/CNN_ConMAE/cnn_conmae_only-weights.pth")
    # _example_state_dict = torch.load(_example_state_dict_path, map_location="cpu")
    #
    # # remove the "model." prefix from the keys
    # _example_state_dict = {
    #     k.replace("model.", ""): v for k, v in _example_state_dict.items()
    # }
    # for k, v in _example_state_dict.items():
    #     print(k, v.shape)
    #
    # model = resenc_l(
    #     mode="segmentation",
    #     input_channels=3,
    #     num_classes=1,
    #     output_channels=1,
    #     deep_supervision=False
    # )
    # _x = torch.randn(4, 3, 64, 64, 64)  # Example input tensor
    # model.load_state_dict(_example_state_dict)  # Load the state dict
    # model.eval()  # Set the model to evaluation mode
    #
    # # print the state dict keys and shapes
    # for k, v in model.state_dict().items():
    #     print(k, v.shape)
    # exit(0)
    #
    # output = model(_x)
    # print(output.shape)  # Should print the shape of the output tensor
    # # For segmentation, the output shape should be (4, 1, 64, 64, 64)
    # del model
    #
    # model = resenc_l(
    #     mode="classification",
    #     input_channels=2,
    #     num_classes=10,  # Example for classification
    #     output_channels=10,
    #     deep_supervision=False
    # )
    # _x = torch.randn(4, 2, 64, 64, 64)  # Example input tensor for classification
    # output = model(_x)
    # print(output.shape)  # Should print the shape of the output tensor for classification, e.g., (4, 10)
    # del model

    model = resenc_l(
        mode="regression",
        input_channels=1,
        num_classes=1,  # Example for regression
        output_channels=1,
        deep_supervision=False
    )
    _x = torch.randn(4, 1, 64, 64, 64)  # Example input tensor for regression
    output = model(_x)
    print(output.shape)  # Should print the shape of the output tensor for regression, e.g., (4, 1)
    del model
