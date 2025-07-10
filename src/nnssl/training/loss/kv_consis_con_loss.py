from typing import Literal
from functools import lru_cache

import torch
import torch.nn.functional as F

from nnssl.training.loss.mse_loss import MAEMSELoss


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
        for i in range(2 * batch_size)
        for j in range(2 * batch_size)
        if i % batch_size != j % batch_size
    ]
    return tuple(
        idxs for idxs in zip(*pairs)
    )  # returns two lists: first and second elements of the pairs


class KVConsisConLoss(torch.nn.Module):

    def __init__(self, device, p=2, epsilon=0.1, out_size=7, sampling_ratio=2):
        """
        Initialize the KVConsisConLoss with the given parameters.

        Args:
            device (torch.device): The device to run the loss on.
            p (int): The norm degree for the consistency loss.
            epsilon (float): A small value to avoid division by zero.
        """
        super(KVConsisConLoss, self).__init__()
        self.p = p
        self.epsilon = epsilon

        self.mse_loss = MAEMSELoss()
        self.log_cosh = LogCoshError(reduction="none")
        self.huber = torch.nn.HuberLoss(reduction="none")

        self.recon_key = "recon"
        self.proj_key = "proj"
        self.latent_key = "proj"

        # create a grid for resampling later
        # ── determine output resolution ──────────────────────────────────────────────
        if isinstance(out_size, int):
            D_out = H_out = W_out = out_size
        else:
            D_out, H_out, W_out = out_size
        self.D_out, self.H_out, self.W_out = D_out, H_out, W_out

        # ── build a base grid in [-1, 1] ────────────────────────────────────────────
        z_lin = torch.linspace(-1, 1, sampling_ratio * D_out, device=device)
        y_lin = torch.linspace(-1, 1, sampling_ratio * H_out, device=device)
        x_lin = torch.linspace(-1, 1, sampling_ratio * W_out, device=device)
        zz, yy, xx = torch.meshgrid(
            z_lin, y_lin, x_lin, indexing="ij"
        )  # (D_out, H_out, W_out)
        self.base_grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(
            0
        )  # (1, D_out, H_out, W_out, 3)

    def align_views(
        self,
        latents: torch.Tensor,
        rel_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aligns the latents based on the relative bounding boxes.

        Args:
            latents (torch.Tensor): The latent representations [b, c, x_p, y_p, z_p].
            rel_bboxes (torch.Tensor): The relative bounding boxes. [b, 6] where each row is (x1, y1, z1, x2, y2, z2)
                and the values are in the range [0, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Aligned latents and bounding boxes.
        """
        B = latents.shape[0]

        # ── prepare per‑sample scale & shift ───────────────────────────────────────
        x1, y1, z1, x2, y2, z2 = rel_bboxes.unbind(dim=-1)  # each (B,)
        # centre in [0,1], size in [0,1]
        cx, cy, cz = (x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5
        sx, sy, sz = (x2 - x1), (y2 - y1), (z2 - z1)

        # convert to shift / scale for [-1,1] space
        shift = torch.stack((2 * cx - 1, 2 * cy - 1, 2 * cz - 1), dim=-1)  # (B, 3)
        scale = torch.stack((sx, sy, sz), dim=-1)  # (B, 3)

        shift = shift.view(B, 1, 1, 1, 3)
        scale = scale.view(B, 1, 1, 1, 3)

        # ── produce the sampling grid ──────────────────────────────────────────────
        sampling_grid = self.base_grid * scale + shift  # (B, D_out, H_out, W_out, 3)

        # ── trilinear ROI‑align via grid_sample ────────────────────────────────────
        aligned_latents = F.grid_sample(
            latents,
            sampling_grid,
            mode="bilinear",  # when 5d input, "bilinear" is equivalent to "trilinear" internally
            padding_mode="border",
            align_corners=True,
        )

        # ── adaptive pool to the ouput size ────────────────────────────────────────
        aligned_latents = F.adaptive_avg_pool3d(
            aligned_latents, (self.D_out, self.H_out, self.W_out)
        )

        return aligned_latents

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
        eps = torch.finfo(model_output[self.recon_key].dtype).eps

        recon_loss_lc = self.log_cosh(model_output[self.recon_key], gt_recon)
        recon_loss_lc = torch.sum(recon_loss_lc * (1 - mask)) / (torch.sum((1 - mask)) + eps)

        recon_loss_huber = self.huber(model_output[self.recon_key], gt_recon)
        recon_loss_huber = torch.sum(recon_loss_huber * (1 - mask)) / (torch.sum((1 - mask)) + eps)

        recon_loss_mse = self.mse_loss(
            model_output[self.recon_key], gt_recon, mask
        )

        # chunk the latents and compute the consistency loss
        pred_latents = model_output[self.proj_key]

        tgt_latents = target[self.latent_key].detach()
        cw_std = torch.std(F.normalize(tgt_latents.mean(dim=(2, 3, 4)), dim=1), dim=(0,), keepdim=True).mean() / (1 / tgt_latents.shape[1] ** 0.5)

        # if latents is 5d tensor, i.e. [b, c, x_p, y_p, z_p], we need to align them for better consistency
        if pred_latents.ndim == 5:
            pred_latents = self.align_views(pred_latents, rel_bboxes)
            tgt_latents = self.align_views(tgt_latents, rel_bboxes)

        b = pred_latents.shape[0] // 2
        tgt_latents = tgt_latents.roll(
            b, 0
        )  # swap the latents. the num_views is hardcoded to 2 for this method

        # attraction_term_lp = torch.norm(pred_latents - tgt_latents, p=self.p, dim=1)
        # attraction_term_lp = torch.mean(attraction_term_lp)

        # neg_idxs_a, neg_idxs_b = get_neg_pairs(b)
        # neg_idxs_a, neg_idxs_b = (
        #     torch.tensor(neg_idxs_a, device=pred_latents.device),
        #     torch.tensor(neg_idxs_b, device=pred_latents.device),
        # )

        # repulsion_terms_lp = torch.norm(
        #     pred_latents[neg_idxs_a] - tgt_latents[neg_idxs_b], p=self.p, dim=1
        # )
        # repulsion_terms_lp = torch.mean(repulsion_terms_lp)

        pred_latents_fg, tgt_latents_fg = F.normalize(pred_latents, dim=1), F.normalize(tgt_latents, dim=1)

        fg_cos_reg = 2 - 2 * (pred_latents_fg * tgt_latents_fg).sum(dim=1).mean()  # already normalized

        # # aggregate the aligned feature maps over the spatial dimensions
        # pred_latents_aa, tgt_latents_aa = pred_latents.mean(dim=(2, 3, 4)), tgt_latents.mean(dim=(2, 3, 4))
        # pred_latents_aa, tgt_latents_aa = F.normalize(pred_latents_aa, dim=1), F.normalize(tgt_latents_aa, dim=1)
        #
        # attract_cos_aa = 2 - 2 * (pred_latents_aa * tgt_latents_aa).sum(dim=1).mean()  # already normalized
        # repel_cos_aa = 2 - 2 * (pred_latents_aa[neg_idxs_a] * tgt_latents_aa[neg_idxs_b]).sum(dim=1).mean()
        #
        # # contrastive_loss_lp = attraction_term_lp / (repulsion_terms_lp + self.epsilon)
        # contrastive_loss_cos = attract_cos_aa / (repel_cos_aa + self.epsilon)

        # loss = recon_loss_huber + contrastive_loss_lp + negative_cosine_regression
        loss = recon_loss_huber + 0.5 * fg_cos_reg

        return {
            "loss": loss,
            "log_cosh": recon_loss_lc,
            "huber": recon_loss_huber,
            "mse": recon_loss_mse,
            "cw_std": cw_std,
            # "cl_cos": contrastive_loss_cos,
            # "aa_pos_cos": attract_cos_aa,
            # "aa_neg_cos": repel_cos_aa,
            "fg_cos_reg": fg_cos_reg,
        }


if __name__ == "__main__":
    pairs = get_neg_pairs(4)
    print(pairs, len(pairs[0]), len(pairs[1]))

    _model_output = {
        "recon": torch.randn(4, 1, 64, 64, 64, requires_grad=True, device="cuda"),
        "proj": torch.randn(4, 256, 20, 20, 20, requires_grad=True, device="cuda"),
    }

    _target = {
        "recon": torch.randn(4, 1, 64, 64, 64, device="cuda"),
        "proj": torch.randn(4, 256, 20, 20, 20, device="cuda"),
    }

    _gt_recon = torch.randn(
        4, 1, 64, 64, 64, device="cuda"
    )  # Ground truth reconstruction

    _rel_bboxes = torch.tensor(
        [
            [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
            [0.3, 0.3, 0.3, 0.7, 0.7, 0.7],
            [0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
        ],
        device="cuda",
    )  # Example relative bounding boxes
    _mask = torch.randint(
        0, 2, (4, 1, 64, 64, 64), device="cuda"
    )  # Random mask for the example

    loss_fn = KVConsisConLoss(torch.device("cuda"), p=2, epsilon=0.1)
    _loss_output = loss_fn(
        model_output=_model_output,
        target=_target,
        gt_recon=_gt_recon,
        rel_bboxes=_rel_bboxes,
        mask=_mask,
    )
    print(_loss_output)
