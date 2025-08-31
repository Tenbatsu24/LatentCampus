import torch
from utils.masking import generate_random_mask


def random_mask(x, mask_ratio, mask_patch_size, mask_token=0):
    mask = generate_random_mask(x, mask_ratio, mask_patch_size, out_type=bool)
    assert isinstance(mask, torch.BoolTensor) or isinstance(
        mask, torch.cuda.BoolTensor
    ), mask.type()
    x[mask] = mask_token
    return x, mask
