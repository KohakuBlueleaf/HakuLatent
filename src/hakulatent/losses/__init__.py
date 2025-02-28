import torch
import torch.nn as nn
from convnext_perceptual_loss import ConvNextType

from .adversarial import AdvLoss
from .perceptual import PerceptualLoss, LPIPSLoss, ConvNeXtPerceptualLoss
from .vq_loss import KeplerQuantizerRegLoss


loss_table = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "gnll": nn.GaussianNLLLoss,
}

def srgb_to_oklab(srgb: torch.Tensor) -> torch.Tensor:
    # Convert to linear RGB space.
    rgb = torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        # Clamping avoids NaNs in backwards pass
        ((torch.clamp(srgb, min=0.04045) + 0.055) / 1.055) ** 2.4
    )

    # Convert RGB to LMS (cone response)
    t_rgb_lms = torch.tensor([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ], dtype=srgb.dtype, device=srgb.device)

    lin_lms = torch.tensordot(rgb, t_rgb_lms, dims=([1], [0]))

    # Cone response cuts off at low light and we rely on rods, but assume a
    # linear response in low light to preserve differentiablity.
    # (2/255) / 12.92, which is roughly in the range of scotopic vision
    # (2e-6 cd/m^2) given a bright 800x600 CRT at 250 cd/m^2.

    # Apply nonlinearity to LMS
    X = 6e-4
    A = (X ** (1/3)) / X

    lms = torch.where(
        lin_lms <= X,
        lin_lms * A,
        # Clamping avoids NaNs in backwards pass
        torch.clamp(lin_lms, min=X) ** (1/3)
    )

    # Convert LMS to Oklab
    t_lms_oklab = torch.tensor([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ], dtype=srgb.dtype, device=srgb.device)

    return torch.tensordot(lms, t_lms_oklab, dims=([1], [0]))

class ReconLoss(nn.Module):
    def __init__(
        self,
        loss_type="mse",
        lpips_net="alex",
        loss_colorspace="srgb",
        convnext_type=None,
        convnext_kwargs={},
        loss_weights={},
    ):
        super(ReconLoss, self).__init__()
        self.loss = loss_table[loss_type]()
        self.loss_weight = loss_weights.get(loss_type, 1.0)
        self.loss_colorspace = loss_colorspace
        if lpips_net is not None:
            self.lpips_loss = LPIPSLoss(lpips_net)
            self.lpips_weight = loss_weights.get("lpips", 1.0)
        else:
            self.lpips_loss = None
        if convnext_type is not None:
            self.convn_loss = ConvNeXtPerceptualLoss(
                model_type=convnext_type, **convnext_kwargs
            )
            self.convn_weight = loss_weights.get("convnext", 1.0)
        else:
            self.convn_loss = None

    def forward(self, x_real, x_recon):
        if isinstance(self.loss, nn.GaussianNLLLoss):
            x_recon, var = torch.split(
                x_recon, (x_real.size(1), x_recon.size(1) - x_real.size(1)), dim=1
            )
            # var = var.expand(-1, x_real.size(1), -1, -1)

        # losses relying on trained networks need to stay as sRGB
        x_real_srgb = x_real
        x_recon_srgb = x_recon

        if self.loss_colorspace == "srgb":
            pass # assumed that pixel data is in sRGB space
        elif self.loss_colorspace == "oklab":
            x_real = srgb_to_oklab(x_real)
            x_recon = srgb_to_oklab(x_recon)
        else:
            raise NotImplementedError

        if isinstance(self.loss, nn.GaussianNLLLoss):
            base = self.loss(x_recon, x_real, torch.abs(var) + 1) * self.loss_weight
        else:
            base = self.loss(x_recon, x_real) * self.loss_weight

        if self.lpips_loss is not None:
            lpips = self.lpips_loss(x_recon_srgb, x_real_srgb)
            base += lpips * self.lpips_weight
        if self.convn_loss is not None:
            convn = self.convn_loss(x_recon_srgb, x_real_srgb)
            base += convn * self.convn_weight
        return base
