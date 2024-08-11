import cv2
import os
import random
import numpy as np
import PIL.Image
import torch


from networks.mat import Generator01
from networks.csmat import Generator02

import dnnlib
import legacy


def inpaint_cv2(image, mask, method="telea", radius=3):

    flags = cv2.INPAINT_NS
    if method == "telea":
        flags = cv2.INPAINT_TELEA
    print(method)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # perform inpainting using OpenCV
    output = cv2.inpaint(image, mask, radius, flags=flags)
    return output


###################################################################################
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


def inpaint_mat(image: np.ndarray, mask: np.ndarray, model_path: str,  method="mat",
                resolution: int = 512, truncation_psi: float = 1.0, 
                noise_mode: str = 'const') -> np.ndarray:
    """
    Inpaint image using pretrained network.

    Args:
        image (np.ndarray): Input image array.
        mask (np.ndarray): Mask image array.
        model_path (str): Path to the pretrained model pickle.
        resolution (int): Resolution of the input image.
        truncation_psi (float): Truncation psi.
        noise_mode (str): Noise mode ('const', 'random', 'none').

    Returns:
        np.ndarray: Inpainted image.
    """
    # pick a random number
    seed = 2048
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda')
    with dnnlib.util.open_url(model_path) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
    net_res = 512 if resolution > 512 else resolution
    if method == "mat":
        G = Generator01(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    elif method == "cs-mat":
        G = Generator02(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    # Normalize input image
    image_data = (torch.from_numpy(image.transpose(2, 0, 1)).float().to(device) / 127.5 - 1).unsqueeze(0)

    # Normalize mask image
    mask_data = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask_data = 1 - torch.from_numpy(mask_data).float().to(device).unsqueeze(0).unsqueeze(0)
    # mask_data = torch.from_numpy(mask_data).float().to(device).unsqueeze(0).unsqueeze(0)

    # Generate latent vector
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    with torch.no_grad():
        output = G(image_data, mask_data, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()

    return output

def ours_inpaint(image, mask, method="mat"):
    model_path01 = 'model/mat-25m(public).pkl'
    model_path02 = 'model/cs-mat-4m2(new-loss).pkl'
    if method == "mat":
        output = inpaint_mat(image, mask, model_path01, method=method)
    elif method == "cs-mat":
        output = inpaint_mat(image, mask, model_path02, method=method)
    return output