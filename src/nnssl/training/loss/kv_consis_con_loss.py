from typing import Literal
from functools import lru_cache

import torch


class LogCoshError(torch.nn.Module):
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "none"):
        """
        Initialize the LogCoshError loss function.
        """
        super(LogCoshError, self).__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'.")

        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LogCoshError loss.

        Args:
            input (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.

        Returns:
            torch.Tensor: The computed Log-Cosh error.
        """
        eps = torch.finfo(input.dtype).eps
        diff = input - target
        loss = torch.log(torch.cosh(diff + eps))
        if self.reduction == "mean":
            return torch.mean(loss)  # (1, )
        elif self.reduction == "sum":
            return torch.sum(loss)  # (1,
        else:
            return loss  # [b, ...] shape, no reduction applied


@lru_cache(maxsize=5)
def get_neg_pairs(
    batch_size: int,
) -> tuple:
    """
    Generate positive and negative masks for the given batch size.

    Args:
        batch_size (int): The size of the batch.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the negative pairs.
    """
    pairs = [
        (i, j)
        for i in range(batch_size)
        for j in range(batch_size)
        if i % batch_size != j % batch_size
    ] + [
        (i, j)
        for i in range(batch_size, 2 * batch_size)
        for j in range(batch_size, 2 * batch_size)
        if i % batch_size != j % batch_size
    ]
    return tuple(
        idxs for idxs in zip(*pairs)
    )  # returns two lists: first and second elements of the pairs


class KVConsisConLoss(torch.nn.Module):

    def __init__(self, p=2, epsilon=0.1):
        """
        Initialize the KVConsisConLoss with the given parameters.

        Args:
            p (int): The norm degree for the consistency loss.
            epsilon (float): A small value to avoid division by zero.
        """
        super(KVConsisConLoss, self).__init__()
        self.p = p
        self.epsilon = epsilon

        self.mae_loss = LogCoshError(reduction="none")
        self.resampling_grid_size = (7, 7, 7)  # Default grid size for resampling

        # create a grid for resampling later
        self.resampling_grid = torch.meshgrid(
            torch.linspace(0, 1, self.resampling_grid_size[0]),
            torch.linspace(0, 1, self.resampling_grid_size[1]),
            torch.linspace(0, 1, self.resampling_grid_size[2]),
            indexing="ij",
        )

    def align_views(self,
        latents: torch.Tensor,
        rel_bboxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aligns the latents based on the relative bounding boxes.

        Args:
            latents (torch.Tensor): The latent representations [b, c, x_p, y_p, z_p].
            rel_bboxes (torch.Tensor): The relative bounding boxes. [b, 6] where each row is (x1, y1, z1, x2, y2, z2)
                and the values are in the range [0, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Aligned latents and bounding boxes.
        """
        b, c, x_p, y_p, z_p = latents.shape
        rel_bboxes = 2 * (rel_bboxes - 0.5)  # Convert to [-1, 1] range
        rel_bboxes = rel_bboxes.view(b, 2, 3)  # Reshape to [b, 2, 3] for easier indexing
        aligned_latents = torch.empty(b, c, *self.resampling_grid_size, device=latents.device)
        pass

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        gt_recon: torch.Tensor,
        rel_bboxes: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the KVConsisConLoss.

        Args:
            model_output (dict[str, torch.Tensor]): The output from the model.
            target (dict[str, torch.Tensor]): The target values.
            gt_recon (torch.Tensor): Ground truth reconstruction.
            abs_bboxes (torch.Tensor): Relative bounding boxes.
            mask (torch.Tensor): Mask to apply to the loss.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Compute the consistency loss
        eps = torch.finfo(model_output["recon"].dtype).eps

        recon_loss = self.mae_loss(model_output["recon"], gt_recon)
        recon_loss = torch.sum(recon_loss * mask) / (torch.sum(mask) + eps)

        # chunk the latents and compute the consistency loss
        pred_latents = model_output["latents"]
        b = pred_latents.shape[0] // 2

        tgt_latents = target["latents"]
        tgt_latents = tgt_latents.roll(b, 0)  # swap the latents

        attraction_term = torch.norm(pred_latents - tgt_latents, p=self.p, dim=1)
        attraction_term = torch.mean(attraction_term)

        neg_idxs_a, neg_idxs_b = get_neg_pairs(b)
        neg_idxs_a, neg_idxs_b = (
            torch.tensor(neg_idxs_a, device=pred_latents.device),
            torch.tensor(neg_idxs_b, device=pred_latents.device),
        )
        repulsion_terms = torch.norm(
            pred_latents[neg_idxs_a] - tgt_latents[neg_idxs_b], p=self.p, dim=1
        )
        repulsion_terms = torch.mean(repulsion_terms)

        loss = attraction_term / (repulsion_terms + self.epsilon) + recon_loss

        return {
            "loss": loss,  # average over the batch
            "recon_loss": recon_loss,
            "attraction_term": attraction_term,
            "repulsion_terms": repulsion_terms,
        }


if __name__ == "__main__":
    pairs = get_neg_pairs(8)
    print(pairs, len(pairs[0]), len(pairs[1]))

    _model_output = {
        "recon": torch.randn(16, 3, 64, 64, 64),
        "latents": torch.randn(16, 320),
    }

    _target = {
        "recon": torch.randn(16, 3, 64, 64, 64),
        "latents": torch.randn(16, 320),
    }

    _gt_recon = torch.randn(16, 3, 64, 64, 64)

    _rel_bboxes = torch.rand(16, 6)  # Example relative bounding boxes
    _mask = torch.ones(16, 1, 64, 64, 64)  # Example mask

    loss_fn = KVConsisConLoss(p=2, epsilon=0.1)
    _loss_output = loss_fn(
        model_output=_model_output,
        target=_target,
        gt_recon=_gt_recon,
        rel_bboxes=_rel_bboxes,
        mask=_mask,
    )
    print(_loss_output)
