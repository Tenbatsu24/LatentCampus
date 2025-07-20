import numpy as np


def random_overlap_tuple(ol: float):
    """
    Generates a random tuple of overlap fractions for three axes that sum to the given total overlap.
    Args:
        ol: float - total overlap fraction (0 < ol < 1)
    Returns:
        tuple of three floats representing the overlap fractions for each axis
        (ol_x, ol_y, ol_z) such that ol_x * ol_y * ol_z = ol
    """
    ln_ol = np.log(ol)
    weights = np.random.dirichlet(np.ones(3))
    return tuple(np.exp(weights * ln_ol))


def make_overlapping_crops_w_bbox(
    batch, patch_size, min_overlap, max_overlap, debug=False
):
    """
    Args:
        batch: np.ndarray of shape (b, c, x, y, z)
        patch_size: tuple (px, py, pz) - size of the crop per axis
        min_overlap: float - minimum overlap fraction (0 < min_overlap < 1)
        max_overlap: float - maximum overlap fraction (0 < max_overlap < 1)
        debug: bool - if True, prints debug information

    Returns:
        crops1: np.ndarray of shape (b, c, px, py, pz)
        crops2: np.ndarray of shape (b, c, px, py, pz)
        bboxes1: np.ndarray of shape (b, 6) - (x1, y1, z1, x2, y2, z2) in crop-relative coords (0-1)
        bboxes2: np.ndarray of shape (b, 6) - same as above for second crop
    """
    b, c, X, Y, Z = batch.shape
    px, py, pz = patch_size
    crops_1 = np.zeros((b, c, px, py, pz), dtype=batch.dtype)
    crops_2 = np.zeros((b, c, px, py, pz), dtype=batch.dtype)
    bboxes1 = np.zeros((b, 6), dtype=np.float32)
    bboxes2 = np.zeros((b, 6), dtype=np.float32)

    for i in range(b):
        # Sample overlap amounts
        # Sample total volume overlap and split into per-axis overlap
        total_overlap = np.random.uniform(low=min_overlap, high=max_overlap)
        overlap_frac = random_overlap_tuple(total_overlap)
        if debug:
            print(overlap_frac)

        overlap = [int(patch_size[d] * overlap_frac[d]) for d in range(3)]
        start_1, start_2 = [], []
        for d, (v, p, o) in enumerate(
            ([X, px, overlap[0]], [Y, py, overlap[1]], [Z, pz, overlap[2]])
        ):
            # Ensure there's room on both sides of the overlap region
            min_center = o // 2 + (p - o) + 1  # off by one to ensure overlap is valid
            max_center = (
                v - (p - o) - (o // 2) - 1
            )  # off by one to ensure overlap is valid

            if max_center <= min_center:
                _start1 = _start2 = 0  # fail-safe case, no valid overlap
            else:
                center = np.random.randint(min_center, max_center + 1)

                _start1 = center + o // 2 - p
                _start2 = center - o // 2

            start_1.append(_start1)
            start_2.append(_start2)

        if debug:
            # Debugging information
            ol_ratio = np.prod(overlap / np.array(patch_size))
            print(f"Overlap: {ol_ratio}, Start1: {start_1}, Start2: {start_2}")
            print(f"{np.prod(overlap)} / {np.prod(patch_size)} = {ol_ratio}")

            import matplotlib.pyplot as plt

            # plot a 3d representation of the two crops by making a cuboid using the start points and patch size
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title(
                f"Batch {i + 1} - Overlap: {overlap}, Start1: {start_1}, Start2: {start_2}"
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            # Create cuboids for the two crops
            for start, color in zip([start_1, start_2], ["red", "blue"]):
                ax.bar3d(
                    start[0],
                    start[1],
                    start[2],
                    patch_size[0],
                    patch_size[1],
                    patch_size[2],
                    color=color,
                    alpha=0.3,
                    edgecolor="k",
                )

            plt.show()
            # exit(0)

        # Extract crops
        crop1 = batch[
            i,
            :,
            start_1[0] : start_1[0] + px,
            start_1[1] : start_1[1] + py,
            start_1[2] : start_1[2] + pz,
        ]
        crop2 = batch[
            i,
            :,
            start_2[0] : start_2[0] + px,
            start_2[1] : start_2[1] + py,
            start_2[2] : start_2[2] + pz,
        ]

        crops_1[i] = crop1
        crops_2[i] = crop2

        # compute the relative bounding boxes in the crop coordinates
        bboxes1[i] = np.array(
            [
                1 - overlap_frac[0],
                1 - overlap_frac[1],
                1 - overlap_frac[2],
                1.0,
                1.0,
                1.0,
            ]
        )

        bboxes2[i] = np.array(
            [
                0.0,
                0.0,
                0.0,
                overlap_frac[0],
                overlap_frac[1],
                overlap_frac[2],
            ]
        )

    return (crops_1, crops_2), (bboxes1, bboxes2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(random_overlap_tuple(0.2))

    print(np.prod(random_overlap_tuple(0.2)))  # 0.2

    _b = 4
    _x, _y, _z = 128, 128, 128
    _xp, _yp, _zp = 64, 64, 64

    _batch = np.random.rand(_b, 1, _x, _y, _z).astype(np.float32)
    _patch_size = (_xp, _yp, _zp)
    _min_ol = 0.4
    _max_ol = 0.8

    (c1, c2), (bb1, bb2) = make_overlapping_crops_w_bbox(
        _batch, _patch_size, _min_ol, _max_ol, debug=True
    )
    print(c1.shape)  # (8, 1, 64, 64, 64)
    print(bb1[0], bb2[0])

    # show the bbox as rectangles for a 2d slice on each axis for all the crops
    _, _axs = plt.subplots(_b, 3 * 2, figsize=(30, 5 * _b))
    for _b_idx in range(_b):
        for _ax_idx, _dim in enumerate((2, 3, 4, 2, 3, 4)):
            ax = _axs[_b_idx, _ax_idx]
            if _ax_idx >= 3:
                _c, _bb = c2, bb2
                color = "blue"
            else:
                _c, _bb = c1, bb1
                color = "red"

            if _dim == 2:
                _full_im = _c[_b_idx, 0, _xp // 2, :, :]
                _bbox = _bb[_b_idx, [0, 1, 3, 4]] * _xp
            elif _dim == 3:
                _full_im = _c[_b_idx, 0, :, _yp // 2, :]
                _bbox = _bb[_b_idx, [0, 2, 3, 5]] * _yp
            else:
                _full_im = _c[_b_idx, 0, :, :, _zp // 2]
                _bbox = _bb[_b_idx, [1, 2, 4, 5]] * _zp
            ax.imshow(np.zeros_like(_full_im), cmap="gray")
            ax.add_patch(
                plt.Rectangle(
                    (_bbox[0], _bbox[1]),
                    _bbox[2] - _bbox[0],
                    _bbox[3] - _bbox[1],
                    edgecolor=color,
                    facecolor="none",
                    lw=2,
                )
            )
            ax.set_title(f"Crop {_b_idx + 1} - Dim {_dim + 1}")

    plt.tight_layout()
    plt.show()
