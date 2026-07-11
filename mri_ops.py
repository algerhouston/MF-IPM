import torch


def fft2c(image):
    """Centered-style MRI FFT wrapper using orthonormal scaling."""
    return torch.fft.fft2(image, dim=(-2, -1), norm="ortho")


def ifft2c(kspace):
    """Inverse MRI FFT wrapper using orthonormal scaling."""
    return torch.fft.ifft2(kspace, dim=(-2, -1), norm="ortho")


def real_image(kspace):
    return ifft2c(kspace).real


def _broadcast_mask(mask, reference):
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(1)
    return mask.to(device=reference.device, dtype=reference.real.dtype)


def apply_kspace_mask(kspace, mask):
    mask = _broadcast_mask(mask, kspace)
    return kspace * mask


def undersample_image(image, mask):
    kspace = fft2c(image)
    return real_image(apply_kspace_mask(kspace, mask))


def data_consistency(image, measured_kspace, mask):
    """Replace predicted k-space samples with measured samples at mask locations."""
    pred_kspace = fft2c(image)
    mask = _broadcast_mask(mask, pred_kspace)
    corrected = pred_kspace * (1.0 - mask) + measured_kspace * mask
    return real_image(corrected)


def complex_to_channels(kspace):
    if not torch.is_complex(kspace):
        return kspace
    return torch.cat([kspace.real, kspace.imag], dim=1)
