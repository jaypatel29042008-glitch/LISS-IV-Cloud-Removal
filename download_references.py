"""
Team Brain Wave — ISRO BAH 2026
Research Paper & Reference Downloader
Downloads PDFs from arXiv and other open-access sources
"""

import urllib.request
import os
import ssl

# Disable SSL verification for downloads (some servers have cert issues)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "references")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAPERS = [
    {
        "name": "DB-CR_Diffusion_Bridge_Cloud_Removal_2025",
        "url": "https://arxiv.org/pdf/2504.03607",
        "description": "Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal (Hu et al., 2025)"
    },
    {
        "name": "DiffCR_Fast_Conditional_Diffusion_2024",
        "url": "https://arxiv.org/pdf/2308.04417",
        "description": "DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal (Zou et al., 2024)"
    },
    {
        "name": "SEN12MS-CR_Dataset_Paper",
        "url": "https://arxiv.org/pdf/2211.02408",
        "description": "SEN12MS-CR: A Dataset for SAR-Optical Multimodal Cloud Removal"
    },
    {
        "name": "UnCRtainTS_Cloud_Removal_2023",
        "url": "https://arxiv.org/pdf/2304.05798",
        "description": "UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series"
    },
    {
        "name": "Pix2Pix_Image_Translation_2017",
        "url": "https://arxiv.org/pdf/1611.07004",
        "description": "Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)"
    },
    {
        "name": "DDPM_Denoising_Diffusion_2020",
        "url": "https://arxiv.org/pdf/2006.11239",
        "description": "Denoising Diffusion Probabilistic Models (Ho et al., 2020)"
    },
]

def download_paper(paper):
    filename = os.path.join(OUTPUT_DIR, f"{paper['name']}.pdf")
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        print(f"  [SKIP] Already exists: {paper['name']}.pdf")
        return True
    
    print(f"  Downloading: {paper['description'][:70]}...")
    try:
        req = urllib.request.Request(
            paper['url'],
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
            data = response.read()
            with open(filename, 'wb') as f:
                f.write(data)
            size_mb = len(data) / (1024 * 1024)
            print(f"  [OK] Saved: {paper['name']}.pdf ({size_mb:.1f} MB)")
            return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    print("=" * 70)
    print("  Team Brain Wave - Research Paper Downloader")
    print("=" * 70)
    print()
    
    success = 0
    for paper in PAPERS:
        if download_paper(paper):
            success += 1
        print()
    
    print(f"Downloaded {success}/{len(PAPERS)} papers to: {OUTPUT_DIR}")
    
    # Also save a reference index
    index_file = os.path.join(OUTPUT_DIR, "PAPER_INDEX.md")
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("# Research Paper Index - Team Brain Wave\n\n")
        f.write("| # | Paper | Source | Status |\n")
        f.write("|---|-------|--------|--------|\n")
        for i, paper in enumerate(PAPERS, 1):
            pdf_exists = os.path.exists(os.path.join(OUTPUT_DIR, f"{paper['name']}.pdf"))
            status = "Downloaded" if pdf_exists else "Pending"
            f.write(f"| {i} | {paper['description']} | [PDF]({paper['url']}) | {status} |\n")
        
        f.write("\n\n## GitHub Repositories\n\n")
        repos = [
            ("DB-CR (Diffusion Bridge Cloud Removal)", "https://github.com/Aayush99000/SAR-Guided-Optical-Cloud-Reconstruction-Diffusion-Bridge-"),
            ("DiffCR (Fast Conditional Diffusion)", "https://github.com/XavierJiezou/DiffCR"),
            ("UnCRtainTS (Uncertainty Cloud Removal)", "https://github.com/PatrickTUM/UnCRtainTS"),
            ("RICE Dataset", "https://github.com/BUPTLdy/RICE_DATASET"),
        ]
        for name, url in repos:
            f.write(f"- [{name}]({url})\n")
        
        f.write("\n\n## Datasets\n\n")
        f.write("- **SEN12MS-CR**: [MediaTUM](https://mediatum.ub.tum.de/1474000) — 122K triplets (SAR+Cloudy+Clear), 100GB+\n")
        f.write("- **RICE Dataset**: [GitHub](https://github.com/BUPTLdy/RICE_DATASET) — 500+736 pairs (512x512)\n")
        f.write("- **ISRO Bhuvan/Bhoonidhi**: [Bhoonidhi Portal](https://bhoonidhi.nrsc.gov.in) — LISS-IV Level-1T GeoTIFF tiles\n")
        f.write("- **Copernicus Sentinel-2 L2A**: [Data Space](https://dataspace.copernicus.eu) — 10m Blue/Green/Red/NIR, 20m SWIR\n")
        f.write("- **Copernicus Sentinel-1 GRD**: [Data Space](https://dataspace.copernicus.eu) — C-band SAR (VV/VH) for cloud-free reference\n")
    
    print(f"\nPaper index saved to: {index_file}")


if __name__ == "__main__":
    main()
