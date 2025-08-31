from io import BytesIO
from math import ceil, floor
from pathlib import Path
from typing import Optional
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


FPS = 4
DPI = 100


def finish():
    wandb.finish()


def update_config(dct):
    wandb.config.update(dct)


def save_tensor(x, name):
    torch.save(x, name)
    wandb.save(name)
    print(f"Saved {name} to wandb")


def get_gif(x, y, y_hat, slice_dim, version_dir, epoch, desc=""):
    assert version_dir is not None
    assert epoch is not None
    path = f"{version_dir}/gifs"
    Path(path).mkdir(parents=False, exist_ok=True)

    figs = get_figs(x, y, y_hat, slice_dim=slice_dim, n=None, desc=desc)
    frames = []

    for fig in figs:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=DPI)
        buf.seek(0)
        frame_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        frame = cv2.imdecode(frame_arr, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.close(fig)
        frames.append(Image.fromarray(frame))

    file_name = f"{path}/epoch_{epoch}.gif"
    # Create gif using some PIL magic
    frames[0].save(file_name, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0)
    gif = wandb.Video(file_name, format="gif", fps=FPS)

    plt.close("all")

    return [gif]


def get_imgs(x, y, y_hat, slice_dim, n, desc="", idx=0):
    figs = get_figs(x, y, y_hat, slice_dim=slice_dim, n=n, desc=desc, batch_idx=idx)

    def wandb_img(fig):
        img = wandb.Image(fig)
        plt.close(fig)
        return img

    plt.close("all")

    return [wandb_img(fig) for fig in figs]


def get_figs(x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, slice_dim: int, n: Optional[int] = None, desc="", batch_idx: int = 0):
    assert len(x.shape) == 5, x.shape  # (B, 1, X, Y, Z)
    assert len(y.shape) == 5, y.shape  # (B, 1, X, Y, Z)
    assert len(y_hat.shape) == 5, y_hat.shape  # (B, 1, X, Y, Z)

    x = x[batch_idx].squeeze().detach().cpu()
    y = y[batch_idx].squeeze().detach().cpu()
    y_hat = y_hat[batch_idx].squeeze().detach().cpu()

    # we might use mixed precision, so we cast to ensure tensor is compatiable with numpy
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    y_hat = y_hat.to(torch.float32)

    figs = []

    if n is None:
        index_range = torch.arange(x.shape[slice_dim])
    else:
        # we take the middle num_png_images images
        middle = x.shape[slice_dim] // 2
        index_range = torch.arange(middle - floor(n / 2), middle + ceil(n / 2))
        assert len(index_range) == n

    for i in index_range:
        xi = torch.index_select(x, slice_dim, i).squeeze()
        yi = torch.index_select(y, slice_dim, i).squeeze()
        yi_hat = torch.index_select(y_hat, slice_dim, i).squeeze()

        if xi.min() == xi.max() and yi.min() == yi.max():
            continue  # slice is empty

        fig = create_image(xi, yi, yi_hat, rec_text=f"i: {i}", desc=desc)
        figs.append(fig)

    return figs


def create_image(xi, yi, yi_hat, rec_text="", desc=""):
    # x: (X, Y)
    # y: (X, Y)
    # y_hat: (X, Y)

    assert len(xi.shape) == 2, xi.shape
    assert len(yi.shape) == 2, yi.shape
    assert len(yi_hat.shape) == 2, yi_hat.shape

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=DPI)

    im = axes[0].imshow(yi, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(xi, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Masked")
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(yi_hat, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Reconstructed {rec_text}")
    fig.colorbar(im, ax=axes[2])

    fig.text(0.5, 0.05, desc, ha="center")

    return fig
