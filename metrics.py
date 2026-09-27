"""
Team Brain Wave — Quantitative Metrics Module
Computes PSNR, SSIM, SAM (Spectral Angle Mapper), and GCI for satellite image quality evaluation.
"""

import numpy as np

def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between ground truth (img1) and reconstructed (img2).
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_val ** 2) / mse)


def compute_ssim_single_channel(x: np.ndarray, y: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes Structural Similarity Index (SSIM) for 2D single-channel arrays.
    """
    K1, K2 = 0.01, 0.03
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2
    
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    sigma_x_sq = np.var(x)
    sigma_y_sq = np.var(y)
    sigma_xy = np.cov(x.flatten(), y.flatten())[0, 1] if x.size > 1 else 0.0
    
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    return float(num / den) if den != 0 else 1.0


def compute_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Computes mean SSIM across all channels.
    Supports (H, W), (H, W, C), or (C, H, W).
    """
    if img1.ndim == 2:
        return compute_ssim_single_channel(img1, img2, max_val)
    
    if img1.shape[0] in [1, 3, 4, 5]: # (C, H, W)
        ssims = [compute_ssim_single_channel(img1[c], img2[c], max_val) for c in range(img1.shape[0])]
    else: # (H, W, C)
        ssims = [compute_ssim_single_channel(img1[..., c], img2[..., c], max_val) for c in range(img1.shape[-1])]
    
    return float(np.mean(ssims))


def compute_sam(img1: np.ndarray, img2: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in radians.
    Measures spectral vector similarity across bands at each pixel location.
    Lower is better (0 = identical spectral shape).
    """
    # Reshape to (C, N)
    if img1.ndim == 3:
        if img1.shape[0] in [3, 4, 5]:
            v1 = img1.reshape(img1.shape[0], -1)
            v2 = img2.reshape(img2.shape[0], -1)
        else:
            v1 = img1.reshape(-1, img1.shape[-1]).T
            v2 = img2.reshape(-1, img2.shape[-1]).T
    else:
        v1 = img1.reshape(1, -1)
        v2 = img2.reshape(1, -1)

    dot_product = np.sum(v1 * v2, axis=0)
    norm1 = np.linalg.norm(v1, axis=0)
    norm2 = np.linalg.norm(v2, axis=0)

    denominator = norm1 * norm2 + eps
    cos_theta = np.clip(dot_product / denominator, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    
    return float(np.mean(angles))


def compute_ndvi(red_band: np.ndarray, nir_band: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalized Difference Vegetation Index (NDVI) = (NIR - Red) / (NIR + Red)
    """
    return (nir_band - red_band) / (nir_band + red_band + eps)


def compute_gci_accuracy(gt_red: np.ndarray, gt_nir: np.ndarray,
                         pred_red: np.ndarray, pred_nir: np.ndarray) -> float:
    """
    Computes Green Cover Index (GCI) recovery accuracy percentage (0-100%).
    """
    ndvi_gt = compute_ndvi(gt_red, gt_nir)
    ndvi_pred = compute_ndvi(pred_red, pred_nir)
    mae = np.mean(np.abs(ndvi_gt - ndvi_pred))
    accuracy = max(0.0, (1.0 - mae)) * 100.0
    return float(accuracy)


if __name__ == "__main__":
    # Sanity check
    ref = np.random.rand(3, 128, 128)
    noisy = ref + np.random.normal(0, 0.05, ref.shape)
    noisy = np.clip(noisy, 0, 1)

    print(f"PSNR: {compute_psnr(ref, noisy):.2f} dB")
    print(f"SSIM: {compute_ssim(ref, noisy):.4f}")
    print(f"SAM:  {compute_sam(ref, noisy):.4f} rad")
    print(f"GCI:  {compute_gci_accuracy(ref[1], ref[2], noisy[1], noisy[2]):.1f}%")
