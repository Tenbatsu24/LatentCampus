from typing import Optional
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice

from yucca.modules.optimization.loss_functions.deep_supervision import (
    DeepSupervisionLoss,
)
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE
from yucca.modules.metrics.training_metrics import F1

from models.supervised_base import BaseSupervisedModel


class SupervisedSegModel(BaseSupervisedModel):
    """
    Supervised model for segmentation tasks.
    Inherits from BaseSupervisedModel and implements segmentation-specific functionality.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        deep_supervision: bool = False,
    ):
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=deep_supervision,
        )

    def _configure_metrics(self, prefix: str):
        """
        Configure segmentation-specific metrics

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of segmentation metrics
        """
        return MetricCollection(
            {
                f"{prefix}/dice": Dice(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                ),
                f"{prefix}/F1": F1(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                    average=None,
                ),
            },
        )

    def _configure_losses(self):
        """
        Configure segmentation-specific loss functions

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        loss_fn_train = DiceCE(soft_dice_kwargs={"apply_softmax": True})
        loss_fn_val = DiceCE(soft_dice_kwargs={"apply_softmax": True})

        if self.deep_supervision:
            loss_fn_train = DeepSupervisionLoss(loss_fn_train, weights=None)

        return loss_fn_train, loss_fn_val

    def compute_metrics(self, metrics, output, target, ignore_index: int = 0):
        """
        Compute segmentation metrics, handling per-class results

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (usually background)

        Returns:
            dict: Dictionary of computed metrics
        """
        metrics = metrics(output, target)
        tmp = {}
        to_drop = []
        for key in metrics.keys():
            if metrics[key].numel() > 1:
                to_drop.append(key)
                for i, val in enumerate(metrics[key]):
                    if not i == ignore_index:
                        tmp[key + "_" + str(i)] = val
        for k in to_drop:
            metrics.pop(k)
        metrics.update(tmp)
        return metrics
