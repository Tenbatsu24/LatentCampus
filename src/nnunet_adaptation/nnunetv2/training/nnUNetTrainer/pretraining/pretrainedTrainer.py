from typing import Literal, Tuple, Union, List

import torch

from torch import nn, autocast
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import isfile, join
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)

from nnunetv2.utilities.get_network_via_name import get_network_from_name
from nnunetv2.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
)
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)

warmup_stages = Literal["warmup_all", "warmup_decoder", "train_all", "train_decoder"]


class PretrainedTrainer(nnUNetTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        use_pretrained_weights: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.use_pretrained_weights = use_pretrained_weights
        if not self.use_pretrained_weights:
            self.initial_lr = 1e-2
        self.adaptation_info = self.plans_manager.plans["pretrain_info"]
        self.training_stage = None
        self.pt_weight_in_ch_mismatch = False

    def print_citations(self):
        cits = self.adaptation_info.get("citations", [])
        if len(cits) > 0:
            all_strings = []
            all_strings.append(
                "\n#######################################################################\nPlease cite the associated papers when using pre-trained weights:\n"
            )
            for cit in sorted(cits, key=lambda x: x["type"]):
                all_strings.append(
                    f"{cit['type']} used '{cit['name']}'. Associated paper(s):"
                )
                for c in cit["apa_citations"]:
                    all_strings.append(c)
                all_strings[-1] += "\n"
            all_strings.append(
                "#######################################################################\n"
            )
            final_string = "\n".join(all_strings)
            self.print_to_log_file(final_string, add_timestamp=False)
        return

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # During `nnUNetv2_preprocess_like_nnssl` we create a new plan that specifies the architecture already.
            #   This plan holds details on how the architecture is supposed to be built.

            self.network = self.build_network_architecture(
                architecture_class_name=self.configuration_manager.network_arch_class_name,
                arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
                arch_init_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
                input_patch_size=self.configuration_manager.patch_size,  # Set in plan to pt_recommended_patchsize
                num_input_channels=self.num_input_channels,
                num_output_channels=self.label_manager.num_segmentation_heads,
                enable_deep_supervision=False,
            ).to(self.device)

            # Load pretrained weights
            if self.use_pretrained_weights:
                assert (
                    "checkpoint_path" in self.adaptation_info
                ), "`checkpoint_path` not found in plans! Can't load weights"
                assert isfile(
                    self.adaptation_info["checkpoint_path"]
                ), f"Pretrained weights path {self.adaptation_info['checkpoint_path']} does not exist!"
                self.network, self.pt_weight_in_ch_mismatch = (
                    self.load_pretrained_weights(
                        self.network,
                        pretrained_weights_path=self.adaptation_info["checkpoint_path"],
                        pt_input_channels=self.adaptation_info["pt_num_in_channels"],
                        downstream_input_channels=self.num_input_channels,
                        pt_input_patchsize=self.adaptation_info["pt_used_patchsize"],
                        downstream_input_patchsize=self.adaptation_info[
                            "pt_recommended_downstream_patchsize"
                        ],
                        pt_key_to_encoder=self.adaptation_info["key_to_encoder"],
                        pt_key_to_stem=self.adaptation_info["key_to_stem"],
                        pt_keys_to_in_proj=tuple(
                            self.adaptation_info["keys_to_in_proj"]
                        ),
                        pt_key_to_lpe=self.adaptation_info["key_to_lpe"],
                    )
                )
                self.print_citations()
                self.print_to_log_file(
                    "Loaded Network from {}".format(
                        self.adaptation_info["checkpoint_path"]
                    )
                )
            else:
                self.print_to_log_file(
                    "You are using a Trainer for fine-tuning but without loading weigts"
                )
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    @staticmethod
    def load_pretrained_weights(
        network: AbstractDynamicNetworkArchitectures,
        pretrained_weights_path: str,
        pt_input_channels: int,
        downstream_input_channels: int,
        pt_input_patchsize: int,
        downstream_input_patchsize: int,
        pt_key_to_encoder: str,
        pt_key_to_stem: str,
        pt_keys_to_in_proj: tuple[str, ...],
        pt_key_to_lpe: str,
    ) -> tuple[nn.Module, bool]:
        """
        Load pretrained weights into the network.
        Per default we only load the encoder and the stem weights. The stem weights are adapted to the number of input channels through repeats.
        The decoder is initialized from scratch.

        :param network: The neural network to load weights into.
        :param pretrained_weights_path: Path to the pretrained weights file.
        :param pt_input_channels: Number of input channels used in the pretrained model.
        :param downstream_input_channels: Number of input channels used during adaptation (currently).
        :param pt_input_patchsize: Patch size used in the pretrained model.
        :param downstream_input_patchsize: Patch size used during adaptation (currently).
        :param pt_key_to_encoder: Key to the encoder in the pretrained model.
        :param pt_key_to_stem: Key to the stem in the pretrained model.

        :return: The network with loaded weights.
        """

        # --------------------------- Technical Description -------------------------- #
        # In this function we want to load the weights in a reliable manner.
        #   Hence we want to load the weights with `strict=False` to guarantee everything is loaded as expected.
        #   To do so, we grab the respective submodules and load the fitting weights into them.
        #   We can do this through `get_submodule` which is a nn.Module function.
        #   However we need to cut-away the prefix of the matching keys to correctly assign weights from both `state_dicts`!
        # Difficulties:
        # 1) Different stem dimensions: When pre-training had only a single input channel, we need to make the shapes fit!
        #    To do so, we utilize repeating the weights N times (N = number of input channels).
        #    Limitation currently we only support this for a single input channel used during pre-training.
        # 2) Different patch sizes: The learned positional embeddings LPe of `Transformer` (Primus) architectures are
        #    patch size dependent. To adapt the weights, we do trilinear interpolation of these weights back to shape.
        # 3) Stem and Encoder merging: Most architectures (Primus, ResidualEncoderUNet derivatives) have
        #    separate `stem` and `encoder` objects. Hence we can separate stem and encoder weight loading easily.
        #    However in the `PlainConvUNet` architecture the encoder contains the stem, so we must make sure
        #    to skip the stem weight loading in the encoder, and then separately load the (repeated) stem weights

        # The following code does this.

        key_to_encoder = (
            network.key_to_encoder
        )  # Key to the encoder in the current network
        key_to_stem = (
            network.key_to_stem
        )  # Key to the stem (beginning) in the current network

        random_init_statedict = network.state_dict()
        pre_train_statedict: dict[str, torch.Tensor] = torch.load(
            pretrained_weights_path, weights_only=True
        )[
            "network_weights"  # Get pre-trained state dict
        ]
        stem_in_encoder = pt_key_to_stem in pre_train_statedict

        # Currently we don't have the logic for interpolating the positional embedding yet.
        pt_weight_in_ch_mismatch = False
        need_to_ignore_lpe = False  # I.e. Learnable positional embedding
        key_to_lpe = getattr(network, "key_to_lpe", None)
        # Check if the current module even uses a learnable positional embedding. If not ignore LPE logic.
        try:
            network.get_submodule(key_to_lpe)
        except AttributeError:
            key_to_lpe = None

        if key_to_lpe is not None:
            # Add interpolation logic for positional embeddings later
            lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
            lpe_in_stem = key_to_lpe.startswith(key_to_stem)
            if pt_input_patchsize != downstream_input_patchsize:
                need_to_ignore_lpe = (
                    True  # LPE shape won't fit -> replace with random init
                )
                #  We actually tested impact of using interpolated LPE and it's basically identical.
                #  So we just ignore it at the moment.
            # Should the LPE be neither in the encoder nor the stem, we don't need to specifically ignore it.
            #   However when the patch sizes are identical, we need to explicitly load it.

        def strip_dot_prefix(s) -> str:
            """Mini func to strip the dot prefix from the keys"""
            if s.startswith("."):
                return s[1:]
            return s

        # ----- Match the keys of pretrained weights to the current architecture ----- #
        if stem_in_encoder:
            encoder_weights = {
                k: v
                for k, v in pre_train_statedict.items()
                if k.startswith(pt_key_to_encoder)
            }
            if downstream_input_channels > pt_input_channels:
                pt_weight_in_ch_mismatch = True
                k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
                vals = (
                    encoder_weights[k_proj].repeat(1, downstream_input_channels, 1, 1)
                ) / downstream_input_channels
                for k in pt_keys_to_in_proj:
                    encoder_weights[k] = vals
            # Fix the path to the weights:
            new_encoder_weights = {
                strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v
                for k, v in encoder_weights.items()
            }
            # --------------------------------- Adapt LPE -------------------------------- #
            if need_to_ignore_lpe:
                if lpe_in_encoder:
                    new_encoder_weights[
                        strip_dot_prefix(key_to_lpe.replace(pt_key_to_encoder, ""))
                    ] = random_init_statedict[key_to_lpe]

            # ------------------------------- Load weights ------------------------------- #
            encoder_module = network.get_submodule(key_to_encoder)
            encoder_module.load_state_dict(new_encoder_weights)
        else:
            encoder_weights = {
                k: v
                for k, v in pre_train_statedict.items()
                if k.startswith(pt_key_to_encoder)
            }
            stem_weights = {
                k: v
                for k, v in pre_train_statedict.items()
                if k.startswith(pt_key_to_stem)
            }
            if downstream_input_channels > pt_input_channels:
                pt_weight_in_ch_mismatch = True
                k_proj = pt_keys_to_in_proj[0] + ".weight"  # Get the projection weights
                vals = (
                    stem_weights[k_proj].repeat(1, downstream_input_channels, 1, 1, 1)
                ) / downstream_input_channels
                for k in pt_keys_to_in_proj:
                    stem_weights[k + ".weight"] = vals
            new_encoder_weights = {
                strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v
                for k, v in encoder_weights.items()
            }
            new_stem_weights = {
                strip_dot_prefix(k.replace(pt_key_to_stem, "")): v
                for k, v in stem_weights.items()
            }
            # --------------------------------- Adapt LPE -------------------------------- #
            if need_to_ignore_lpe:
                if (
                    lpe_in_stem
                ):  # Since stem not in encoder we need to take care of lpe in it here
                    new_stem_weights[
                        strip_dot_prefix(key_to_lpe.replace(key_to_stem, ""))
                    ] = random_init_statedict[key_to_lpe]
                elif lpe_in_encoder:
                    new_encoder_weights[
                        strip_dot_prefix(key_to_lpe.replace(key_to_encoder, ""))
                    ] = random_init_statedict[key_to_lpe]
                else:
                    pass

            # ------------------------------- Load weights ------------------------------- #
            encoder_module = network.get_submodule(key_to_encoder)
            encoder_module.load_state_dict(new_encoder_weights)
            stem_module = network.get_submodule(key_to_stem)
            stem_module.load_state_dict(new_stem_weights)

        if not need_to_ignore_lpe and key_to_lpe is not None:
            # Load the positional embedding weights
            lpe_weights = {
                k: v
                for k, v in pre_train_statedict.items()
                if k.startswith(pt_key_to_lpe)
            }
            assert (
                len(lpe_weights) == 1
            ), f"Found multiple lpe weights, but expect only a single tensor. Got {list(lpe_weights.keys())}"
            network.get_parameter(key_to_lpe).data = list(lpe_weights.values())[0]
            # ------------------------------- Load weights ------------------------------- #

        # Theoretically we don't need to return the network, but we do it anyway.
        return network, pt_weight_in_ch_mismatch

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        input_patch_size: tuple[int, int, int],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> AbstractDynamicNetworkArchitectures:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        if architecture_class_name in [
            "PlainConvUNet",
            "ResidualEncoderUNet",
        ] or architecture_class_name.endswith(("PlainConvUNet", "ResidualEncoderUNet")):
            network = get_network_from_plans(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                allow_init=True,
                deep_supervision=enable_deep_supervision,
            )
        elif architecture_class_name in [
            "PrimusS",
            "PrimusM",
            "PrimusL",
            "PrimusB",
            "ResEncL",
        ]:
            network = get_network_from_name(
                architecture_class_name,
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                input_patchsize=input_patch_size,
                allow_init=True,
                deep_supervision=False,
            )
        else:
            raise NotImplementedError(
                "Unknown architecture class name: {}".format(architecture_class_name)
            )
        return network

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.SGD(
                params,
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer, self.initial_lr, self.warmup_duration_whole_net
            )
            self.print_to_log_file(
                f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.99,
                    nesterov=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer,
                self.initial_lr,
                self.num_epochs,
                self.warmup_duration_whole_net,
            )
            self.print_to_log_file(
                f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        pass

    def on_train_epoch_start(self):
        """Steers the learning rate schedule used during fine-tuning."""
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

    def get_stage(self):
        if self.current_epoch < self.warmup_duration_whole_net:
            stage = "warmup_all"
        else:
            stage = "train"
        return stage

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        We need to overwrite that entire function because we need to fiddle the correct optimizer in between
        loading the checkpoint and applying the optimizer states. Yuck.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(
                filename_or_checkpoint, map_location=self.device, weights_only=False
            )  # will be changed soon
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith(
                "module."
            ):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else self.inference_allowed_mirroring_axes
        )

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        self.optimizer, self.lr_scheduler = self.configure_optimizers(self.get_stage())

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class PretrainedTrainer_Primus(PretrainedTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        use_pretrained_weights: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, use_pretrained_weights, device
        )
        # Can be overriden to train same architecture from scratch.
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.use_pretrained_weights = use_pretrained_weights
        if not self.use_pretrained_weights:
            self.initial_lr = 3e-4
        self.adaptation_info = self.plans_manager.plans["pretrain_info"]

    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params,
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
                fused=True,
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer, self.initial_lr, self.warmup_duration_whole_net
            )
            self.print_to_log_file(
                f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer,
                self.initial_lr,
                self.num_epochs,
                self.warmup_duration_whole_net,
            )
            self.print_to_log_file(
                f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}"
            )
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        pass


class PretrainedTrainer_150ep(PretrainedTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        use_pretrained_weights: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, use_pretrained_weights, device
        )
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150

class PretrainedTrainer_150ep_50i(PretrainedTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        use_pretrained_weights: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, use_pretrained_weights, device
        )
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150
        self.num_iterations_per_epoch = 50  # lin increase whole network


class PretrainedTrainer_100ep_50i(PretrainedTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        use_pretrained_weights: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, use_pretrained_weights, device
        )
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 10  # lin increase whole network
        self.num_epochs = 100
        self.num_iterations_per_epoch = 50  # lin increase whole network


class PretrainedTrainer_Primus_150ep(PretrainedTrainer_Primus):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        use_pretrained_weights: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, use_pretrained_weights, device
        )
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150  # lin increase whole network
