import torch

from nnssl.training.loss.contrastive_loss import NTXentLoss
from nnssl.training.loss.kv_consis_con_loss import LogCoshError


class XentMAELoss(torch.nn.Module):
    """
    Cross-Entropy loss for Masked Autoencoder (MAE) with contrastive learning.
    This loss function is designed to work with the outputs of a MAE model.
    """

    def __init__(self, device, batch_size, temperature=0.5, similarity_function="cosine"):
        """
        Initializes the XentMAELoss with the specified parameters.
        """
        super(XentMAELoss, self).__init__()

        self.contrastive = NTXentLoss(
            batch_size=batch_size,
            temperature=temperature,
            similarity_function=similarity_function,
            device=device,
        )
        self.log_cosh = LogCoshError(reduction="none")
        self.huber = torch.nn.HuberLoss(reduction="none")

        self.recon_key = "recon"
        self.proj_key = "proj"

        self.device = device

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        gt_recon: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Forward pass for the XentMAELoss.
        Computes the contrastive loss and reconstruction loss based on the model output.

        Args:
            model_output (dict[str, torch.Tensor]): Model outputs containing 'recon' and 'proj'.
            gt_recon (torch.Tensor): Ground truth reconstruction tensor.
            mask (torch.Tensor): Mask tensor indicating valid positions for loss computation.

        Returns:
            torch.Tensor: Computed loss value.
        """
        eps = torch.finfo(model_output[self.recon_key].dtype).eps

        recon = model_output[self.recon_key]
        proj = model_output[self.proj_key]

        proj_1, proj_2 = proj.chunk(2, dim=0)

        # Compute contrastive loss
        contrastive_loss, _ = self.contrastive(proj_1, proj_2)

        # Compute reconstruction loss
        recon_loss_lc = self.log_cosh(recon, gt_recon)
        recon_loss_lc = torch.sum(recon_loss_lc * mask) / (torch.sum(mask) + eps)

        recon_loss_mse = self.huber(recon, gt_recon)
        recon_loss_mse = torch.sum(recon_loss_mse * mask) / (torch.sum(mask) + eps)


        # Combine losses
        loss = contrastive_loss + recon_loss_lc + recon_loss_mse

        return {
            "loss": loss,
            "cl_NTXent": contrastive_loss,
            "log_cosh": recon_loss_lc,
            "huber": recon_loss_mse,
        }


if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    recon_output = torch.randn(2 * batch_size, 1, 64, 64).to(device)
    proj_output = torch.randn(2 * batch_size, 128).to(device)
    gt_recon = torch.randn(2 * batch_size, 1, 64, 64).to(device)
    mask = torch.ones(2 * batch_size, 1, 64, 64).to(device)

    model_output = {
        "recon": recon_output,
        "proj": proj_output
    }

    loss_fn = XentMAELoss(device=device, batch_size=batch_size)
    loss_dict = loss_fn(model_output=model_output, gt_recon=gt_recon, mask=mask)

    print(loss_dict)
