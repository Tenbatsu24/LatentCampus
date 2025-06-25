import torch
from torch import autocast

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    BaseMAETrainer,
)
from nnssl.utilities.helpers import dummy_context


class BaseKVConsisTrainer(BaseMAETrainer):
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

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            # del data
            l = self.loss(output, data, mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        mask = self.mask_creation(
            self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            l = self.loss(output, data, mask)

        return {"loss": l.detach().cpu().numpy()}
