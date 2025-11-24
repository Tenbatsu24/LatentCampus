import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import nn
from einops import rearrange

from nnssl.training.loss.mse_loss import MAEMSELoss


class DINOLoss(nn.Module):
    def __init__(
        self,
        teacher_temp=0.04,
        student_temp=0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, n_iterations=3):
        """
        Fully numerically-stable Sinkhorn-Knopp in log-domain.
        Equivalent to original DINO implementation but immune to overflow.
        Input:
            teacher_output: [batch, prototypes]
        Output:
            Q: [batch, prototypes] assignment matrix
        """
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        B_local, K = teacher_output.shape
        B = B_local * world_size

        # ---------------------------------------------------------------
        # 1. Convert teacher logits -> logQ using temperature
        #    logQ = log( exp(teacher_output / T) )
        #         = teacher_output / T
        # ---------------------------------------------------------------
        log_Q = teacher_output / self.teacher_temp

        # switch to K x B notation (transpose)
        log_Q = log_Q.t()  # [K, B_local]

        # ---------------------------------------------------------------
        # 2. Global normalization: sum(Q) = 1  (in log domain)
        # ---------------------------------------------------------------
        # compute global logsumexp
        local_logZ = torch.logsumexp(log_Q, dim=(0, 1))
        if dist.is_initialized():
            # Reduce log-sum-exp correctly:
            # we must sum the EXP of each worker's contribution
            # so we convert back from logspace safely
            max_logZ = local_logZ.clone()
            dist.all_reduce(max_logZ, op=dist.ReduceOp.MAX)

            # sum(exp(logZ - max)) across workers
            exp_sum = torch.exp(local_logZ - max_logZ)
            dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM)

            # final global logZ
            logZ = max_logZ + torch.log(exp_sum)
        else:
            logZ = local_logZ

        log_Q = log_Q - logZ  # now sum(exp(log_Q)) == 1

        # ---------------------------------------------------------------
        # 3. Sinkhorn iterations (alternating row / column normalization)
        # ---------------------------------------------------------------
        for _ in range(n_iterations):
            # ---- Row normalization: rows sum to 1/K ----
            log_row_sum = torch.logsumexp(log_Q, dim=1, keepdim=True)

            if dist.is_initialized():
                # correct global reduce in log-domain
                max_r = log_row_sum.clone()
                dist.all_reduce(max_r, op=dist.ReduceOp.MAX)
                exp_r = torch.exp(log_row_sum - max_r)
                dist.all_reduce(exp_r, op=dist.ReduceOp.SUM)
                log_row_sum = max_r + torch.log(exp_r)

            # subtract log_row_sum AND log(K)
            log_Q = log_Q - log_row_sum - math.log(K)

            # ---- Column normalization: columns sum to 1/B ----
            log_col_sum = torch.logsumexp(log_Q, dim=0, keepdim=True)

            # single-process is safe, no allreduce needed here
            log_Q = log_Q - log_col_sum - math.log(B)

        # ---------------------------------------------------------------
        # 4. Final: convert back to normal domain safely
        #    Now values are normalized and safe for exp()
        # ---------------------------------------------------------------
        Q = torch.exp(log_Q)  # [K, B_local]
        Q = Q / Q.sum(dim=0, keepdim=True)  # Make sure the columns sum to 1 to be an assignment

        return Q.t()  # return [batch, prototypes]

    def forward(self, student_logits, teacher_logits, ignore_diagonal=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_logits: [student crops * batch, prototypes]
        teacher_logits:  [teacher crops * batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        # 1) Normalize teacher with Sinkhorn
        teacher_probs = self.sinkhorn_knopp_teacher(teacher_logits)

        # 2) Stack into [num_crops, batch, K]
        student_logits = torch.stack(torch.chunk(student_logits, 2, dim=0))
        teacher_probs = torch.stack(torch.chunk(teacher_probs, 2, dim=0))

        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape

        # 3) Student log-softmax for loss
        log_student = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)

        # -------------------------
        # Loss computation
        # -------------------------
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", log_student, teacher_probs)
            loss = loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", log_student, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            loss = loss.sum() / (B * student_crops * teacher_crops - B * min_st)

        # -------------------------
        # Produce logits & labels for accuracy
        # -------------------------
        if not ignore_diagonal:
            # All student-teacher crop pairs
            # Repeat student logits for each teacher crop and vice versa
            student_logits_expanded = student_logits.unsqueeze(1).expand(-1, teacher_crops, -1, -1)  # [s, t, B, K]
            teacher_labels_expanded = teacher_probs.argmax(dim=-1).unsqueeze(0).expand(student_crops, -1, -1)  # [s, t, B]

            # Flatten to [s*t*B, K] and [s*t*B]
            logits = student_logits_expanded.reshape(-1, K)
            labels = teacher_labels_expanded.reshape(-1)
        else:
            # All pairs except diagonal (s == t)
            # Create mask to exclude diagonal
            mask = ~torch.eye(student_crops, teacher_crops, dtype=torch.bool, device=student_logits.device)

            student_logits_expanded = student_logits.unsqueeze(1).expand(-1, teacher_crops, -1, -1)  # [s, t, B, K]
            teacher_labels_expanded = teacher_probs.argmax(dim=-1).unsqueeze(0).expand(student_crops, -1, -1)  # [s, t, B]

            # Apply mask and flatten
            logits = student_logits_expanded[mask].reshape(-1, K)
            labels = teacher_labels_expanded[mask].reshape(-1)

        return loss, logits, labels


class DinoConsisLoss(torch.nn.Module):

    def __init__(
        self,
        device,
        out_size=12,
        sampling_ratio=2,
        recon_weight=1.0,
        cos_reg_weight=0.5,
        ntxent_weight=0.1,
        fine_grained_contrastive: bool = False,
        fine_grained_cosine_regression: bool = False,
    ):
        """
        Initialize the KVConsisConLoss with the given parameters.

        Args:
            device (torch.device): The device to run the loss on.
            out_size (int or tuple[int, int, int]): The output size for the aligned latents.
            sampling_ratio (int): The ratio for sampling the output size.
            recon_weight (float): Weight for the reconstruction loss.
            cos_reg_weight (float): Weight for the finegrained cosine similarity loss.
            ntxent_weight (float): Weight for the NT-Xent loss.
            fine_grained_contrastive (bool): Whether to use fine-grained contrastive loss.
        """
        super(DinoConsisLoss, self).__init__()

        self.mse_loss = MAEMSELoss()
        self.huber = torch.nn.HuberLoss(reduction="none")
        self.fine_grained_contrastive = (
            fine_grained_contrastive  # whether to use fine-grained contrastive loss
        )
        self.fine_grained_cosine_regression = fine_grained_cosine_regression

        self.recon_key = "recon"
        self.latent_key = "latent"
        self.patch_dino_key = "patch_latent"
        self.image_proj_pred_key = "proj_pred"

        self.contrastive_loss = DINOLoss()

        self.recon_weight = recon_weight
        self.patch_weight = cos_reg_weight
        self.global_weight = ntxent_weight

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
        self.base_grid = torch.stack((zz, yy, xx), dim=-1).unsqueeze(
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

        recon_loss_huber = self.huber(model_output[self.recon_key], gt_recon)
        recon_loss_huber = torch.sum(recon_loss_huber * (1 - mask)) / (
            torch.sum((1 - mask)) + eps
        )

        # chunk the latents and compute the consistency loss
        pred_latents_fg = model_output[self.patch_dino_key]
        tgt_latents_fg = target[self.patch_dino_key].detach()

        with torch.no_grad():
            var_denom = 1 / model_output[self.latent_key].shape[1]
            cw_std = torch.std(
                F.normalize(model_output[self.latent_key].detach(), dim=1, eps=eps), dim=0
            ).mean()
            cw_std = cw_std / (var_denom**0.5)

        # if latents is 5d tensor, i.e. [b, c, z_p, x_p, y_p], we need to align them for better consistency
        if pred_latents_fg.ndim == 5:
            pred_latents_fg = self.align_views(pred_latents_fg, rel_bboxes)
            tgt_latents_fg = self.align_views(tgt_latents_fg, rel_bboxes)

        b = pred_latents_fg.shape[0] // 2
        # swap the latents. the num_views is hardcoded to 2 for this method

        pred_latents_fg, tgt_latents_fg = F.normalize(
            pred_latents_fg, dim=1, eps=eps
        ), F.normalize(tgt_latents_fg, dim=1, eps=eps)

        if self.fine_grained_contrastive:
            patch_loss, logits, labels = self.contrastive_loss(
                rearrange(pred_latents_fg, "b c x y z -> (b x y z) c"),
                rearrange(
                    tgt_latents_fg, "b c x y z -> (b x y z) c"
                ),  # swapped assignment already done
            )
            acc = logits.argmax(dim=1).eq(labels).sum() / labels.size(0)
        elif self.fine_grained_cosine_regression:
            tgt_latents_fg = tgt_latents_fg.roll(b, 0)
            patch_loss = (
                2 - 2 * (pred_latents_fg * tgt_latents_fg).sum(dim=1).mean()
            )  # already normalized
            acc = torch.tensor(0.0, device=patch_loss.device, dtype=patch_loss.dtype)
        else:
            tgt_latents_fg = tgt_latents_fg.roll(b, 0)
            # Flatten into [B, N, C]
            _x_p = rearrange(pred_latents_fg, "b c x y z -> b (x y z) c")  # [B, N, C]
            _y_p = rearrange(tgt_latents_fg, "b c x y z -> b (x y z) c")  # [B, N, C]

            # Compute Gram matrices for all batches in parallel
            G_x = torch.bmm(_x_p.transpose(1, 2), _x_p)  # [B, C, C]
            G_y = torch.bmm(_y_p.transpose(1, 2), _y_p)  # [B, C, C]

            # Compute Gram regularization (mean squared difference)
            patch_loss = torch.mean((G_x - G_y) ** 2)
            acc = torch.tensor(0.0, device=patch_loss.device, dtype=patch_loss.dtype)

        pred_latents_aa, tgt_latents_aa = (
            model_output[self.image_proj_pred_key],
            target[self.image_proj_pred_key].detach(),
        )
        tgt_latents_aa = tgt_latents_aa.roll(b, 0)
        global_loss = 2 - 2 * (
            F.normalize(pred_latents_aa, dim=-1, eps=eps) *
            F.normalize(tgt_latents_aa, dim=-1, eps=eps)
        ).sum(dim=1).mean()

        loss = (
            self.recon_weight * recon_loss_huber
            + self.patch_weight * patch_loss
            + self.global_weight * global_loss
        )

        return {
            "loss": loss,
            "huber": recon_loss_huber,
            "cw_std": cw_std,
            "global_loss": global_loss,
            "acc": acc,
            "patch_loss": patch_loss,
        }


if __name__ == "__main__":
    # _get_correlated_mask(4 * 5 * 5 * 5, torch.device("cuda"), using_teacher=True, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model_output = {
        "recon": torch.randn(8, 1, 64, 64, 64, requires_grad=True, device=device),
        "patch_latent": torch.randn(8, 2048, 16, 16, 16, requires_grad=True, device=device),
        "proj_pred": torch.randn(8, 2048, requires_grad=True, device=device),
        "latent": torch.randn(8, 1120, requires_grad=True, device=device),
    }

    _target = {
        "patch_latent": torch.randn(8, 2048, 16, 16, 16, device=device),
        "proj_pred": torch.randn(8, 2048, device=device),
    }

    _gt_recon = torch.randn(
        8, 1, 64, 64, 64, device=device
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
        device=device,
    )  # Example relative bounding boxes
    _mask = torch.randint(
        0, 2, (8, 1, 64, 64, 64), device=device
    )  # Random mask for the example

    loss_fn = DinoConsisLoss(
        device,
        out_size=5,
        fine_grained_contrastive=True,
        fine_grained_cosine_regression=False,
        recon_weight=1.0,
        cos_reg_weight=1.0,
        ntxent_weight=1.0,
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
