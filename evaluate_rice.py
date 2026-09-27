"""
Team Brain Wave — Step 5 Roadmap Implementation: RICE Benchmark Evaluator
Evaluates baseline U-Net & DB-CR Diffusion Bridge on RICE1 (thin cloud) and RICE2 (thick cloud) datasets.
Computes PSNR, SSIM, SAM, GCI metrics and exports side-by-side visual comparison grids.
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from metrics import compute_psnr, compute_ssim, compute_sam, compute_gci_accuracy
from baseline_unet import RICEDataset, CloudUNet
from band_synthesis import BandSynthesizer
from diffusion_bridge_liss4 import DB_CR_DiffusionBridge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
RICE_DIR = os.path.join(PROJECT_DIR, "RICE_DATASET")
EVAL_OUT_DIR = os.path.join(PROJECT_DIR, "data", "evaluation_results")

os.makedirs(EVAL_OUT_DIR, exist_ok=True)

def run_evaluation(max_samples=20):
    print("=" * 70)
    print("  Team Brain Wave — Step 5 Evaluation on RICE Benchmark")
    print("=" * 70)
    
    if not os.path.exists(RICE_DIR):
        print(f"[FAIL] RICE dataset not found at {RICE_DIR}")
        return

    # Load datasets
    rice1 = RICEDataset(RICE_DIR, 'RICE1')
    rice2 = RICEDataset(RICE_DIR, 'RICE2')
    
    print(f"RICE1 (thin cloud) samples available: {len(rice1)}")
    print(f"RICE2 (thick cloud) samples available: {len(rice2)}")
    
    # Initialize models
    unet_model = CloudUNet(3, 3)
    unet_model.eval()
    
    synthesizer = BandSynthesizer()
    dbcr_model = DB_CR_DiffusionBridge(5, 3, timesteps=20)
    dbcr_model.eval()
    
    results = {'RICE1': [], 'RICE2': []}
    
    for ds_name, dataset in [('RICE1', rice1), ('RICE2', rice2)]:
        print(f"\nEvaluating {ds_name} (Testing {min(max_samples, len(dataset))} samples)...")
        
        psnr_list, ssim_list, sam_list, gci_list = [], [], [], []
        
        for i in range(min(max_samples, len(dataset))):
            cloud_tensor, gt_tensor, fname = dataset[i]
            
            c_np = cloud_tensor.numpy() # (3, H, W)
            gt_np = gt_tensor.numpy()   # (3, H, W)
            
            # Step 2: Band Synthesis (Green, Red, NIR -> 5 bands)
            five_band_input = synthesizer.synthesize_5band_stack(c_np) # (5, H, W)
            five_band_tensor = torch.from_numpy(five_band_input).unsqueeze(0) # (1, 5, H, W)
            
            # Step 3.2: DB-CR Inference
            cloud_mask = torch.from_numpy((np.mean(c_np, axis=0) > 0.45).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                pred_tensor = dbcr_model.sample_20step(five_band_tensor, cloud_mask)
            
            pred_np = pred_tensor.squeeze(0).cpu().numpy()
            
            # Compute metrics
            p = compute_psnr(gt_np, pred_np)
            s = compute_ssim(gt_np, pred_np)
            sam = compute_sam(gt_np, pred_np)
            gci = compute_gci_accuracy(gt_np[1], gt_np[2], pred_np[1], pred_np[2])
            
            psnr_list.append(p)
            ssim_list.append(s)
            sam_list.append(sam)
            gci_list.append(gci)
            
            # Save visual comparison for sample 0 and 5
            if i in [0, 5]:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(np.transpose(c_np, (1, 2, 0)))
                axes[0].set_title(f"Cloudy Input ({fname})")
                axes[0].axis('off')
                
                axes[1].imshow(np.squeeze(cloud_mask.numpy()))
                axes[1].set_title("Synthesized Mask")
                axes[1].axis('off')
                
                axes[2].imshow(np.transpose(pred_np, (1, 2, 0)))
                axes[2].set_title(f"DB-CR Restored\nPSNR: {p:.1f}dB | SSIM: {s:.3f}")
                axes[2].axis('off')
                
                axes[3].imshow(np.transpose(gt_np, (1, 2, 0)))
                axes[3].set_title("Ground Truth")
                axes[3].axis('off')
                
                plt.tight_layout()
                fig_path = os.path.join(EVAL_OUT_DIR, f"eval_comparison_{ds_name}_{i}.png")
                plt.savefig(fig_path, dpi=150)
                plt.close()
                print(f"  [SAVED VISUAL] {fig_path}")

        mean_p = float(np.mean(psnr_list))
        mean_s = float(np.mean(ssim_list))
        mean_sam = float(np.mean(sam_list))
        mean_gci = float(np.mean(gci_list))
        
        results[ds_name] = {
            'PSNR': mean_p,
            'SSIM': mean_s,
            'SAM': mean_sam,
            'GCI': mean_gci
        }
        
        print(f"  Results for {ds_name}:")
        print(f"    PSNR: {mean_p:.2f} dB")
        print(f"    SSIM: {mean_s:.4f}")
        print(f"    SAM:  {mean_sam:.4f} rad")
        print(f"    GCI:  {mean_gci:.1f}%")

    print("\n" + "=" * 70)
    print("  SUMMARY EVALUATION TABLE")
    print("=" * 70)
    print(f"{'Dataset':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'SAM (rad)':<10} | {'GCI (%)':<8}")
    print("-" * 55)
    for ds_name, m in results.items():
        print(f"{ds_name:<10} | {m['PSNR']:<10.2f} | {m['SSIM']:<8.4f} | {m['SAM']:<10.4f} | {m['GCI']:<8.1f}")
    print("=" * 70)
    print(f"Evaluation artifacts saved to: {EVAL_OUT_DIR}")

if __name__ == "__main__":
    run_evaluation(max_samples=10)
