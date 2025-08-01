from typing import Literal
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange

from torch import nn

from nnssl.training.loss.mse_loss import MAEMSELoss


def get_warmup_values(
    current_iter: int,
    total_warmup_iters: int = 250 * 10,
    center_momentum_init: float = 0.9,
    center_momentum_final: float = 0.99,
    temperature_init: float = 0.2,
    temperature_final: float = 0.07,
):
    """
    Linearly warm up center momentum and decay temperature.
    """
    progress = min(current_iter / total_warmup_iters, 1.0)

    # Linear interpolation
    center_momentum = center_momentum_init + progress * (center_momentum_final - center_momentum_init)
    temperature = temperature_init + progress * (temperature_final - temperature_init)

    return center_momentum, temperature


def sinkhorn_knopp(Q, n_iter=3, epsilon=1e-6):
    with torch.no_grad():
        Q = Q.exp()
        Q /= Q.sum()

        K, B = Q.size()
        r = torch.ones(K, device=Q.device) / K
        c = torch.ones(B, device=Q.device) / B

        for _ in range(n_iter):
            Q *= (r / (Q.sum(dim=1, keepdim=True) + epsilon))
            Q *= (c / (Q.sum(dim=0, keepdim=True) + epsilon))
        return Q.T  # (B, K)


# Correlated mask with extra positive offsets when using_teacher = True
@lru_cache(maxsize=5)
def _get_correlated_mask(b, device, using_teacher, verbose=False):
    eye = torch.eye(2 * b, device=device, dtype=torch.uint8)
    shifted = eye.roll(-b, dims=1)
    mask = eye + shifted

    if using_teacher:
        l_pos = eye.roll(-b // 2, dims=1)
        r_pos = eye.roll(b // 2, dims=1)
        mask = mask + l_pos + r_pos

    mask = (1 - mask).bool()  # invert to get negatives

    if verbose:
        import matplotlib.pyplot as plt
        plt.imshow(mask.detach().cpu().numpy(), interpolation="nearest")
        plt.title("Correlated Mask")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return mask


class NTXentLoss(nn.Module):
    def __init__(
        self,
        device,
        temperature=0.5,
        similarity_function: Literal["cosine", "dot"] = "cosine",
        using_teacher=False,
        teacher_norm: Literal["sinkhorn", "sharpen", None] = None,
        sharpen_temperature=0.07,
        sinkhorn_iters=3,
        center_momentum=0.9,
        embedding_dim=2048,  # needed for centering buffer
    ):
        super().__init__()
        self.temperature = temperature
        self.similarity_function = similarity_function
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.using_teacher = using_teacher
        self.teacher_norm = teacher_norm
        self.sharpen_temperature = sharpen_temperature
        self.sinkhorn_iters = sinkhorn_iters

        if teacher_norm == "sharpen":
            self.register_buffer("center", torch.zeros(1, embedding_dim, device=device))
            self.center_momentum = center_momentum

    def _similarity(self, x, y):
        return x @ y.T

    def _normalize_teacher(self, zjs):
        if self.teacher_norm is None:
            return zjs
        elif self.teacher_norm == "sinkhorn":
            return sinkhorn_knopp(zjs.T, n_iter=self.sinkhorn_iters, epsilon=torch.finfo(zjs.dtype).eps).T
        elif self.teacher_norm == "sharpen":
            zjs_centered = zjs - self.center
            p = F.softmax(zjs_centered / self.sharpen_temperature, dim=-1)
            # update EMA center
            new_center = p.mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + new_center * (1 - self.center_momentum)
            return p
        else:
            raise ValueError(f"Unknown teacher_norm: {self.teacher_norm}")

    def get_logits(self, zis, zjs):
        b = zis.size(0)
        device = zis.device

        # Normalize teacher outputs (zjs)
        zjs = self._normalize_teacher(zjs)

        reps = torch.cat([zjs, zis], dim=0)  # (2B, D)
        sim = self._similarity(reps, reps)  # (2B, 2B)

        mask = _get_correlated_mask(b, device, self.using_teacher)
        l_pos = torch.diag(sim, b)
        r_pos = torch.diag(sim, -b)
        positives = torch.cat([l_pos, r_pos]).view(2 * b, 1)

        negatives = sim[mask].view(2 * b, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        labels = torch.zeros(2 * b, dtype=torch.long, device=device)
        return logits, labels

    def forward(self, zis, zjs):
        b = zis.size(0)
        logits, labels = self.get_logits(zis, zjs)
        loss = self.criterion(logits, labels)
        accuracy = (logits.argmax(dim=1) == 0).float().mean().item()
        return loss / (2 * b), accuracy


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


class AlignedMAELoss(torch.nn.Module):

    def __init__(
        self,
        device,
        out_size=7,
        sampling_ratio=2,
        recon_weight=1.0,
        fg_cos_weight=0.5,
        ntxent_weight=0.1,
        do_variance_normalisation=False,
        fine_grained_contrastive: bool = False,
        teacher_normalisation: bool = True,
    ):
        """
        Initialize the KVConsisConLoss with the given parameters.

        Args:
            device (torch.device): The device to run the loss on.
            out_size (int or tuple[int, int, int]): The output size for the aligned latents.
            sampling_ratio (int): The ratio for sampling the output size.
            recon_weight (float): Weight for the reconstruction loss.
            fg_cos_weight (float): Weight for the finegrained cosine similarity loss.
            ntxent_weight (float): Weight for the NT-Xent loss.
            do_variance_normalisation (bool): Whether to apply variance normalization.
            fine_grained_contrastive (bool): Whether to use fine-grained contrastive loss.
        """
        super(AlignedMAELoss, self).__init__()

        self.mse_loss = MAEMSELoss()
        self.log_cosh = LogCoshError(reduction="none")
        self.huber = torch.nn.HuberLoss(reduction="none")
        self.do_variance_normalisation = do_variance_normalisation
        self.fine_grained_contrastive = fine_grained_contrastive  # whether to use fine-grained contrastive loss

        self.recon_key = "recon"
        self.proj_key = "proj"
        self.latent_key = "proj"
        self.image_latent_key = "image_latent"

        self.current_iter = 0
        self.contrastive_loss = NTXentLoss(
            device,
            temperature=0.5, similarity_function="cosine", using_teacher=True,
            teacher_norm="sharpen" if teacher_normalisation else None,
            embedding_dim=2048, sharpen_temperature=0.2, center_momentum=0.9
        )

        self.recon_weight = recon_weight
        self.fg_cos_weight = fg_cos_weight
        self.ntxent_weight = ntxent_weight

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
            aligned_latents, (self.W_out, self.H_out, self.D_out)
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
        recon_loss_lc = torch.sum(recon_loss_lc * (1 - mask)) / (
                torch.sum((1 - mask)) + eps
        )

        recon_loss_huber = self.huber(model_output[self.recon_key], gt_recon)
        recon_loss_huber = torch.sum(recon_loss_huber * (1 - mask)) / (
                torch.sum((1 - mask)) + eps
        )

        recon_loss_mse = self.mse_loss(model_output[self.recon_key], gt_recon, mask)

        # chunk the latents and compute the consistency loss
        pred_latents_fg = model_output[self.proj_key]
        tgt_latents_fg = target[self.latent_key].detach()

        var_denom = 1 / target[self.proj_key].shape[1]
        cw_std = torch.std(
            F.normalize(target[self.proj_key].detach(), dim=1, eps=eps),
            dim=(0, 2, 3, 4),
        ).mean()
        cw_std = cw_std / (var_denom ** 0.5)

        # if latents is 5d tensor, i.e. [b, c, x_p, y_p, z_p], we need to align them for better consistency
        if pred_latents_fg.ndim == 5:
            pred_latents_fg = self.align_views(pred_latents_fg, rel_bboxes)
            tgt_latents_fg = self.align_views(tgt_latents_fg, rel_bboxes)

        b = pred_latents_fg.shape[0] // 2
        # swap the latents. the num_views is hardcoded to 2 for this method
        tgt_latents_fg = tgt_latents_fg.roll(b, 0)

        pred_latents_fg, tgt_latents_fg = F.normalize(
            pred_latents_fg, dim=1, eps=eps
        ), F.normalize(tgt_latents_fg, dim=1, eps=eps)

        if self.fine_grained_contrastive:
            fg_cos_reg, var = self.contrastive_loss(
                rearrange(pred_latents_fg, "b c x y z -> (b x y z) c"),
                rearrange(tgt_latents_fg, "b c x y z -> (b x y z) c"),  # swapped assignment already done
            )
            var = torch.tensor(var, dtype=torch.float, device=pred_latents_fg.device)
        else:
            fg_cos_reg = (
                    2 - 2 * (pred_latents_fg * tgt_latents_fg).sum(dim=1).mean()
            )  # already normalized
            var = torch.var(pred_latents_fg, dim=(0, 2, 3, 4), unbiased=False).mean() / var_denom
            # if self.do_variance_normalisation:
            #     fg_cos_reg_n = fg_cos_reg / (
            #             var + eps
            #     )
            # else:
            #     fg_cos_reg_n = fg_cos_reg

        pred_latents_aa, tgt_latents_aa = (
            model_output[self.image_latent_key],
            target[self.image_latent_key].detach(),
        )
        tgt_latents_aa = tgt_latents_aa.roll(b, 0)

        contrastive_loss, acc = self.contrastive_loss(
            F.normalize(pred_latents_aa, dim=1, eps=eps),
            F.normalize(
                tgt_latents_aa.detach(), dim=1, eps=eps
            ),  # already swapped assignments
        )

        loss = (
                self.recon_weight * recon_loss_huber
                + self.fg_cos_weight * fg_cos_reg
                + self.ntxent_weight * contrastive_loss
        )

        if self.do_variance_normalisation and not self.fine_grained_contrastive:
            loss = loss - var

        # if requires gradient, then we update the warmup values
        if self.training:
            self.current_iter += 1
            center_momentum, temperature = get_warmup_values(self.current_iter)
            self.contrastive_loss.center_momentum = center_momentum
            self.contrastive_loss.temperature = temperature

        return {
            "loss": loss,
            "log_cosh": recon_loss_lc,
            "huber": recon_loss_huber,
            "mse": recon_loss_mse,
            "cw_std": cw_std,
            "ntxent": contrastive_loss,
            "acc": torch.tensor(acc, dtype=torch.float, device=loss.device),
            "fg_cos_reg": fg_cos_reg,
            "var": var,
        }


if __name__ == "__main__":
    # _get_correlated_mask(4 * 5 * 5 * 5, torch.device("cuda"), using_teacher=True, verbose=True)

    _model_output = {
        "recon": torch.randn(8, 1, 64, 64, 64, requires_grad=True, device="cuda"),
        "proj": torch.randn(8, 2048, 16, 16, 16, requires_grad=True, device="cuda"),
        "image_latent": torch.randn(8, 2048, requires_grad=True, device="cuda"),
    }

    _target = {
        "proj": torch.randn(8, 2048, 20, 20, 20, device="cuda"),
        "image_latent": torch.randn(8, 2048, device="cuda"),
    }

    _gt_recon = torch.randn(
        8, 1, 64, 64, 64, device="cuda"
    )  # Ground truth reconstruction

    _rel_bboxes = torch.tensor(
        [
            [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
            [0.3, 0.3, 0.3, 0.7, 0.7, 0.7],
            [0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
            [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
            [0.3, 0.3, 0.3, 0.7, 0.7, 0.7],
            [0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
        ],
        device="cuda",
    )  # Example relative bounding boxes
    _mask = torch.randint(
        0, 2, (8, 1, 64, 64, 64), device="cuda"
    )  # Random mask for the example

    loss_fn = AlignedMAELoss(
        torch.device("cuda"), out_size=5, do_variance_normalisation=False, fine_grained_contrastive=True,
        recon_weight=1.0, fg_cos_weight=0.5, ntxent_weight=0.0, teacher_normalisation=True
    )
    loss_fn.train(True)

    _loss_output = loss_fn(
        model_output=_model_output,
        target=_target,
        gt_recon=_gt_recon,
        rel_bboxes=_rel_bboxes,
        mask=_mask,
    )
    print(_loss_output)
