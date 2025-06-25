import torch
from torch import autocast

from nnssl.training.nnsslTrainer.masked_image_modeling.evaMAETrainer import (
    EvaMAETrainer,
)
from nnssl.utilities.helpers import dummy_context


class BaseKVConsisEvaTrainer(EvaMAETrainer):
    """
    Base class for Key-Value Consistency EVA Trainer.
    This class inherits from EvaMAETrainer and is designed to handle
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseKVConsisEvaTrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)

    def get_validation_transforms(self):
        """
        Returns the validation transforms for the model.
        This method is overridden to provide specific validation transforms.
        """
        return super().get_validation_transforms()

    def get_training_transforms(
        self,
        patch_size,
        rotation_for_DA,
        mirror_axes,
        do_dummy_2d_data_aug,
        order_resampling_data=3,
        order_resampling_seg=1,
        use_mask_for_norm=True,
    ):
        """
        Returns the training transforms for the model.
        This method is overridden to provide specific training transforms.
        """
        return super().get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data,
            order_resampling_seg,
            use_mask_for_norm,
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast for CUDA device
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # Forward pass with PatchDropout
            output, keep_indices = self.network(data)
            mask = self.create_mask(
                keep_indices, self.config_plan.patch_size, self.vit_patch_size
            )
            # Calculate loss considering kept patches
            l = self.loss(output, data, mask)

        # Backward pass and optimization
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        # Autocast for CUDA device
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            # Forward pass with PatchDropout
            output, keep_indices = self.network(data)
            mask = self.create_mask(
                keep_indices, self.config_plan.patch_size, self.vit_patch_size
            )
            # Calculate loss considering kept patches
            l = self.loss(output, data, mask)

        return {"loss": l.detach().cpu().numpy()}
