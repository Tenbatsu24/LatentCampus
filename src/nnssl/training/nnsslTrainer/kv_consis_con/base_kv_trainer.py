import copy
from typing import Union, Tuple, List
from typing_extensions import override

import torch
import numpy as np

from torch import autocast

from nnssl.architectures.consis_arch import ConsisMAE
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    BaseMAETrainer,
)
from nnssl.utilities.helpers import dummy_context
from nnssl.ssl_data.dataloading.kv_consis_con_transform import KVConsisTransform


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

        # Default initial patch size, can be overridden in get_dataloaders
        self.initial_patch_size = (256, 256, 256)
        self.total_batch_size = 2
        self.initial_lr = 1e-3
        self.num_epochs = 250
        self.teacher = None
        self.teacher_mom = 0.995  # Momentum for the teacher model update
        self.config_plan.patch_size = (160, 160, 160)  # Default patch size for KV Consis Eva

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.kv_consis_con_loss import KVConsisConLoss

        # Create the loss function
        return KVConsisConLoss(
            device=self.device,
            p=2,
            epsilon=0.1,
        )

    @override
    def build_architecture_and_adaptation_plan(
        self,
        config_plan,
        num_input_channels: int,
        num_output_channels: int,
        *args,
        **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=False
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan

    def get_validation_transforms(self):
        """
        Returns the validation transforms for the model.
        This method is overridden to provide specific validation transforms.
        """
        return KVConsisTransform(
            train="none",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=self.config_plan.patch_size,
        )

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
    ):
        """
        Returns the training transforms for the model.
        This method is overridden to provide specific training transforms.
        """
        return KVConsisTransform(
            train="train",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=patch_size,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            order_resampling_data=order_resampling_data,
            order_resampling_seg=order_resampling_seg,
            border_val_seg=border_val_seg,
            use_mask_for_norm=use_mask_for_norm,
        )

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically.
        """
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        self.initial_patch_size = initial_patch_size
        self.print_to_log_file("Initial patch size: {}".format(initial_patch_size))

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        return self.make_generators(initial_patch_size, tr_transforms, val_transforms)

    def on_train_start(self):
        if self.teacher is None:
            # create a deep copy of the network to use as a teacher
            self.teacher = copy.deepcopy(self.network)
            self.teacher = self.teacher.to(self.device)
            self.teacher = (
                self.teacher.eval()
            )  # set the teacher to eval mode and not training
            for param in self.teacher.parameters():
                param.requires_grad = False
        super().on_train_start()

    def ema(self, teacher_model, student_model, update_bn=False):
        mom = self.teacher_mom

        if mom == 0.0:
            # if the momentum is 0, we just copy the student model to the teacher model
            teacher_model.load_state_dict(student_model.state_dict())
            return

        for p_s, p_t in zip(student_model.parameters(), teacher_model.parameters()):
            p_t.data = mom * p_t.data + (1 - mom) * p_s.data

        if not update_bn:
            return  # update BN stat buffers if required
        for (n_s, m_s), (n_t, m_t) in zip(
            student_model.named_modules(), teacher_model.named_modules()
        ):
            if isinstance(m_s, torch.nn.modules.batchnorm._NormBase) and n_s == n_t:
                m_t.running_mean.data = (
                    mom * m_t.running_mean.data + (1 - mom) * m_s.running_mean.data
                )
                m_t.running_var.data = (
                    mom * m_t.running_var.data + (1 - mom) * m_s.running_var.data
                )

    def shared_step(self, batch: dict, is_train: bool = True) -> dict:
        """
        Shared step for both training and validation.
        This method is overridden to provide specific shared step logic.
        """
        data, bboxes = batch["all_crops"], batch["rel_bboxes"]

        data = data.to(self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)

        with torch.no_grad():
            teacher_output = self.teacher(data)

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            2 * self.batch_size, self.config_plan.patch_size, self.mask_percentage
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

        if is_train:
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
            with torch.no_grad() if not is_train else dummy_context():
                output = self.network(masked_data)
                # del data
                loss_dict = self.loss(
                    model_output=output,
                    target=teacher_output,
                    gt_recon=data,
                    rel_bboxes=bboxes,
                    mask=mask,
                )
                l = loss_dict["loss"]

        if is_train:
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

            # update the teacher network with momentum of 0.95
            with torch.no_grad():
                self.ema(self.teacher, self.network, update_bn=False)

        return {k: v.detach().cpu().numpy() for k, v in loss_dict.items()}

    def train_step(self, batch: dict) -> dict:
        return self.shared_step(batch, is_train=True)

    def validation_step(self, batch: dict) -> dict:
        return self.shared_step(batch, is_train=False)


class KVConsis128Trainer(BaseKVConsisTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the KVConsis128Trainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.config_plan.patch_size = (128, 128, 128)


class KVConsis128SimSiamTrainer(KVConsis128Trainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseKVConsisTrainerSimSiam with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.0

    def build_architecture_and_adaptation_plan(
        self,
        config_plan,
        num_input_channels: int,
        num_output_channels: int,
        *args,
        **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=True
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan


class KVConsis128SimSiamBNTrainer(KVConsis128Trainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseKVConsisTrainerSimSiam with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.0

    def build_architecture_and_adaptation_plan(
        self,
        config_plan,
        num_input_channels: int,
        num_output_channels: int,
        *args,
        **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=True
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan


class KVConsis128SimSiamBNNoConTrainer(KVConsis128Trainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseKVConsisTrainerSimSiam with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.0
