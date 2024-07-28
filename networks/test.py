import os
import random
import numpy as np
import PIL.Image
import torch
import cv2

from networks.mat import Generator
import legacy
import dnnlib


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def read_image(image_path):
    with open(image_path, 'rb') as f:
        image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis]  # HW => HWC
        image = np.repeat(image, 3, axis=2)
    image = image.transpose(2, 0, 1)  # HWC => CHW
    image = image[:3]
    return image


def hello():
    print("Hello World")

def inpaint_mat(image_path: str, mask_path: str, model_path: str, resolution: int = 512, truncation_psi: float = 1.0, noise_mode: str = 'const') -> np.ndarray:
    """
    Inpaint image using pretrained network.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        model_path (str): Path to the pretrained model pickle.
        resolution (int): Resolution of the input image.
        truncation_psi (float): Truncation psi.
        noise_mode (str): Noise mode ('const', 'random', 'none').

    Returns:
        np.ndarray: Inpainted image.
    """
    seed = 240  # pick a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda')
    with dnnlib.util.open_url(model_path) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    # Read input image
    image_data = read_image(image_path)
    image_data = (torch.from_numpy(image_data).float().to(device) / 127.5 - 1).unsqueeze(0)

    # Read mask image
    mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    mask_data = torch.from_numpy(mask_data).float().to(device).unsqueeze(0).unsqueeze(0)

    # Generate latent vector
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    with torch.no_grad():
        output = G(image_data, mask_data, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()

    return output


if __name__ == "__main__":
    model_path = '/home/tphat/Documents/Project/Inpainting/model/org-network-snapshot-000120_4m92.pkl'
    image_path = '/home/tphat/Documents/Project/Inpainting/test_images/CelebA-HQ/images/test2.png'
    mask_path = '/home/tphat/Documents/Project/Inpainting/test_images/CelebA-HQ/masks/mask1.png'
    outdir = '/home/tphat/Documents/Project/Inpainting/output'

    image = inpaint_mat(image_path, mask_path, model_path)
    PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/mine3.png')
