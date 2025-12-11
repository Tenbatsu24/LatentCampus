import math
import random

import torch
import numpy as np


def complete_mask_randomly(mask, num_masking_patches):
    shape = mask.shape
    m2 = mask.flatten()
    num_to_add = num_masking_patches - m2.sum()
    if num_to_add > 0:
        to_add = np.random.choice(np.where(~m2)[0], size=num_to_add, replace=False)
        m2[to_add] = True
    return m2.reshape(shape)


def masked_subset(masks, ratio=0.1):
    """
    masks: (B, HWD) boolean
    ratio: fraction of tokens allowed (e.g. 0.1)
    returns: (B, HWD) boolean mask
    """
    B, HWD = masks.shape
    max_keep = int(ratio * HWD)

    # random scores for masked positions; -inf for unmasked
    rand_scores = torch.rand(B, HWD, device=masks.device)
    rand_scores = rand_scores.masked_fill(~masks, float("-inf"))

    # top-k selection per row
    # topk cannot take per-row k, so use max_keep for all
    topk_vals, topk_idx = rand_scores.topk(max_keep, dim=1)

    # Build the output mask initialized to false
    out = torch.zeros_like(masks, dtype=torch.bool)

    # Scatter valid positions
    out = torch.scatter(
        out, dim=1, index=topk_idx, src=torch.ones_like(masks, dtype=torch.bool)
    )
    out = torch.where(masks, out, torch.zeros_like(masks, dtype=torch.bool))

    return out


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        spatial_dims=3,
        num_masking_patches=None,
        min_num_patches=0,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=3.33,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * spatial_dims
        self.depth, self.height, self.width = input_size

        self.num_patches = self.depth * self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.depth,
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches or -1,
            self.num_masking_patches or -1,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.depth, self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            r1 = math.exp(random.uniform(*self.log_aspect_ratio))
            r2 = math.exp(random.uniform(*self.log_aspect_ratio))

            d = int(round((target_area / (r1 * r2)) ** (1 / 3)))
            h = int(round(r1 * d))
            w = int(round(r2 * d))

            if w < self.width and h < self.height and d < self.depth:
                depth = random.randint(0, self.depth - d)
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[
                    depth : depth + d, top : top + h, left : left + w
                ].sum()
                # Overlap
                if 0 < (h * w * d) - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            for k in range(depth, depth + d):
                                if mask[d, i, j] == 0:
                                    mask[d, i, j] = 1
                                    delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            if self.max_num_patches is not None:
                max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return complete_mask_randomly(mask, num_masking_patches)


def generate_mc_progressive_masks(
    mask_generator,
    number_of_samples,
    num_patch,
    device,
    mask_prob_buckets=(0.0, 0.125, 0.25, 0.375),
    per_bucket_density=None,
):
    num_buckets = len(mask_prob_buckets)
    num_samples_per_crop = number_of_samples // 2
    assert (
        number_of_samples % 2 == 0
    ), "Need the number of samples to be divisble by 2 to use this"

    if per_bucket_density is None:
        per_bucket_density = np.ones(len(mask_prob_buckets)) / num_buckets
    else:
        per_bucket_density = np.array(per_bucket_density, dtype=np.float32)
        per_bucket_density /= float(sum(per_bucket_density))

    masks_per_bucket = [
        int(prob_bucket * num_samples_per_crop) for prob_bucket in per_bucket_density
    ]
    diff = num_samples_per_crop - sum(masks_per_bucket)
    masks_per_bucket[-1] += diff  # correct for non-divisible

    masks_per_bucket = [2 * per_bucket for per_bucket in masks_per_bucket]

    if isinstance(num_patch, tuple):
        num_tokens = math.prod(num_patch)
    else:
        num_tokens = num_patch**3

    masks = [np.empty(1) for _ in range(number_of_samples)]
    bucket_indices = []
    start = 0
    for bucket_idx, p in enumerate(mask_prob_buckets):
        num_in_bucket = masks_per_bucket[bucket_idx]
        half = num_in_bucket // 2

        indices = list(range(start, start + half)) + list(
            range(num_samples_per_crop + start, num_samples_per_crop + start + half)
        )
        start = start + half

        for sample_idx in indices:
            masks[sample_idx] = mask_generator(num_masking_patches=int(p * num_tokens))

        bucket_indices.append(indices)

    bucket_indices = [
        torch.tensor(bucket, device=device, dtype=torch.long)
        for bucket in bucket_indices
    ]

    masks = np.stack(masks, dtype=bool)
    masks = torch.from_numpy(masks).flatten(1, -1)
    masks = masks.to(device)

    return masks, bucket_indices


def main(num_patches):
    import matplotlib.pyplot as plt

    # Initialize the masking generator
    mask_gen = MaskingGenerator(
        input_size=num_patches,  # 16x16 grid
        num_masking_patches=None,  # Total patches to mask
        min_num_patches=0,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=3.3,
    )

    print(f"Masking Generator: {mask_gen}")
    print(f"Grid size: {mask_gen.get_shape()}")
    print(f"Total patches: {mask_gen.num_patches}")

    # Generate multiple masks to visualize
    num_masks = 16
    # min_max_prob = (0.1, 0.5)
    masks, b_idx = generate_mc_progressive_masks(
        mask_gen, num_masks, num_patches, "cpu", per_bucket_density=(1, 1, 2, 2)
    )
    masked_token_masks = masked_subset(masks, ratio=0.1)

    print(
        masks.shape, masked_token_masks.shape, masks.sum(-1), masked_token_masks.sum(-1)
    )

    # shuffle_indices = torch.cat(b_idx, dim=0)
    #
    # inv_idxs = torch.empty_like(shuffle_indices)
    # inv_idxs[shuffle_indices] = torch.arange(
    #     shuffle_indices.size(0), device=shuffle_indices.device
    # )
    # print(shuffle_indices, inv_idxs)

    for i in range(num_masks):
        mask = masks[i]
        print(f"\nMask {i + 1}:")
        print(f"  Masked patches: {mask.sum()} / {mask_gen.num_patches}")
        print(f"  Mask ratio: {mask.sum() / mask_gen.num_patches:.2%}")

    for depth_idx in range(num_patches[0]):

        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(24, 20))
        axes = axes.flatten()

        for idx, (mask, mt_mask, ax) in enumerate(zip(masks, masked_token_masks, axes)):
            # Create visualization
            img = np.zeros((mask_gen.height, mask_gen.width, 3))
            mask = mask.reshape(num_patches)[depth_idx]
            mt_mask = mt_mask.reshape(num_patches)[depth_idx]

            # Unmasked patches: white (1, 1, 1)
            # Masked patches: blue (0.2, 0.4, 0.8)
            img[~mask] = [1, 1, 1]  # White for unmasked
            img[mask] = [0.2, 0.4, 0.8]  # Blue for masked
            img[mt_mask] = [0.8, 0.4, 0.2]

            # Display the mask
            ax.imshow(img, interpolation="nearest")
            ax.set_title(
                f"Mask {idx + 1}: {mask.sum()}/{mask_gen.num_patches} patches masked"
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # Add grid lines
            ax.set_xticks(np.arange(-0.5, mask_gen.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, mask_gen.height, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", size=0)

            # Add patch count annotations
            for i in range(mask_gen.height):
                for j in range(mask_gen.width):
                    color = "white" if mask[i, j] else "black"
                    ax.text(
                        j,
                        i,
                        f"{i * mask_gen.width + j}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=6,
                    )

        plt.tight_layout()
        plt.show()

    # Additional visualization: Show the mask generation process
    print("\n" + "=" * 60)
    print("Demonstrating mask completion:")
    print("=" * 60)


if __name__ == "__main__":
    main(num_patches=(4, 16, 16))
    # _mask_gen = MaskingGenerator(input_size=(16, 16))
    # _masks = generate_masks(_mask_gen, number_of_samples=196, num_patch=14)
    # print(_masks.shape)
