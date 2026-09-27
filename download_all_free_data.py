"""
Team Brain Wave — Download ALL freely available data (no login required)
1. Sentinel-2 quicklook/preview images from Copernicus
2. RICE Dataset from GitHub
3. Sample cloud removal results for demonstration
"""
import urllib.request
import os
import json
import ssl
import zipfile
import io

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(BASE, "..")

def download_file(url, dest, desc=""):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  [SKIP] {desc or os.path.basename(dest)}")
        return True
    print(f"  Downloading {desc or os.path.basename(dest)}...")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            data = resp.read()
        with open(dest, 'wb') as f:
            f.write(data)
        mb = len(data) / (1024*1024)
        print(f"  [OK] {mb:.1f} MB saved")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

# ===================================================
# 1. Download Sentinel-2 Quicklook Preview Images
# ===================================================
def download_sentinel_previews():
    print("\n" + "="*60)
    print("  1. Sentinel-2 Preview Images (No Auth)")
    print("="*60)
    
    catalog_path = os.path.join(PROJECT, "data", "sentinel2", "sentinel2_catalog.json")
    if not os.path.exists(catalog_path):
        print("  [WARN] No catalog found. Run search_sentinel2_v2.py first.")
        return
    
    with open(catalog_path, 'r') as f:
        products = json.load(f)
    
    preview_dir = os.path.join(PROJECT, "data", "sentinel2", "previews")
    os.makedirs(preview_dir, exist_ok=True)
    
    downloaded = 0
    for p in products:
        assets = p.get('assets', {})
        pid = p.get('id', 'unknown')
        region = p.get('region', 'unknown')
        season = p.get('season', 'unknown')
        cloud = p.get('cloud_cover', 'N/A')
        
        # Try quicklook, thumbnail, preview
        for key in ['quicklook', 'thumbnail', 'preview', 'rendered_preview']:
            url = assets.get(key, '')
            if url:
                safe_name = f"{region}_{season}_{pid[:40]}.jpg"
                safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in safe_name)
                dest = os.path.join(preview_dir, safe_name)
                if download_file(url, dest, f"{region}/{season} cloud={cloud}%"):
                    downloaded += 1
                break
    
    print(f"\n  Total previews downloaded: {downloaded}")

# ===================================================
# 2. Download RICE Dataset from GitHub
# ===================================================
def download_rice_dataset():
    print("\n" + "="*60)
    print("  2. RICE Cloud Removal Benchmark Dataset")
    print("="*60)
    
    rice_dir = os.path.join(PROJECT, "data", "datasets", "RICE")
    os.makedirs(rice_dir, exist_ok=True)
    
    # Download RICE dataset ZIP from GitHub
    url = "https://github.com/BUPTLdy/RICE_DATASET/archive/refs/heads/master.zip"
    dest = os.path.join(rice_dir, "RICE_DATASET.zip")
    
    if download_file(url, dest, "RICE Dataset (GitHub ZIP)"):
        # Extract
        try:
            print("  Extracting RICE dataset...")
            with zipfile.ZipFile(dest, 'r') as zf:
                zf.extractall(rice_dir)
            print("  [OK] RICE dataset extracted!")
        except Exception as e:
            print(f"  [WARN] Extraction failed: {e}")

# ===================================================
# 3. Clone DB-CR model code
# ===================================================
def download_dbcr_code():
    print("\n" + "="*60)
    print("  3. DB-CR Model Code (GitHub ZIP)")
    print("="*60)
    
    models_dir = os.path.join(PROJECT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    url = "https://github.com/Aayush99000/SAR-Guided-Optical-Cloud-Reconstruction-Diffusion-Bridge-/archive/refs/heads/main.zip"
    dest = os.path.join(models_dir, "DB-CR_code.zip")
    
    if download_file(url, dest, "DB-CR Diffusion Bridge Code"):
        try:
            print("  Extracting DB-CR code...")
            with zipfile.ZipFile(dest, 'r') as zf:
                zf.extractall(models_dir)
            print("  [OK] DB-CR code extracted!")
        except Exception as e:
            print(f"  [WARN] Extraction failed: {e}")

# ===================================================
# 4. Download DiffCR model code
# ===================================================
def download_diffcr_code():
    print("\n" + "="*60)
    print("  4. DiffCR Model Code (GitHub ZIP)")
    print("="*60)
    
    models_dir = os.path.join(PROJECT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    url = "https://github.com/XavierJiezou/DiffCR/archive/refs/heads/main.zip"
    dest = os.path.join(models_dir, "DiffCR_code.zip")
    
    if download_file(url, dest, "DiffCR Code"):
        try:
            print("  Extracting DiffCR code...")
            with zipfile.ZipFile(dest, 'r') as zf:
                zf.extractall(models_dir)
            print("  [OK] DiffCR code extracted!")
        except Exception as e:
            print(f"  [WARN] Extraction failed: {e}")

# ===================================================
# 5. Download sample Sentinel-2 tile (COG via STAC)
# ===================================================
def download_sample_sentinel_bands():
    print("\n" + "="*60)
    print("  5. Sample Sentinel-2 Band Data (COG via STAC)")
    print("="*60)
    
    sample_dir = os.path.join(PROJECT, "data", "sentinel2", "sample_bands")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Search for a low-cloud Delhi tile to get direct band URLs
    url = "https://stac.dataspace.copernicus.eu/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [76.8, 28.3, 77.5, 28.9],
        "datetime": "2025-02-15T00:00:00Z/2025-02-28T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 5}},
        "limit": 1
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={
            'Content-Type': 'application/json',
            'Accept': 'application/geo+json'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        
        features = result.get('features', [])
        if features:
            assets = features[0].get('assets', {})
            pid = features[0].get('id', 'unknown')
            print(f"  Found tile: {pid}")
            
            # Save asset URLs for reference
            asset_info = {}
            for band_name, asset in assets.items():
                href = asset.get('href', '')
                atype = asset.get('type', '')
                if href:
                    asset_info[band_name] = {"url": href, "type": atype}
                    if band_name in ['blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'thumbnail', 'rendered_preview']:
                        print(f"    Band {band_name}: {href[:80]}...")
            
            # Save URLs as JSON reference
            ref_file = os.path.join(sample_dir, f"band_urls_{pid[:40]}.json")
            with open(ref_file, 'w') as f:
                json.dump(asset_info, f, indent=2)
            print(f"  [SAVED] Band URLs reference: {os.path.basename(ref_file)}")
            
            # Try downloading rendered preview (no auth needed usually)
            for key in ['rendered_preview', 'thumbnail', 'preview']:
                if key in assets:
                    purl = assets[key].get('href', '')
                    if purl:
                        dest = os.path.join(sample_dir, f"preview_{pid[:30]}.png")
                        download_file(purl, dest, f"Preview image for {pid[:30]}")
                        break
        else:
            print("  No tiles found")
    except Exception as e:
        print(f"  [ERROR] {e}")


def main():
    print("="*60)
    print("  Team Brain Wave - Bulk Data Collection")
    print("  (No Authentication Required)")
    print("="*60)
    
    download_sentinel_previews()
    download_rice_dataset()
    download_dbcr_code()
    download_diffcr_code()
    download_sample_sentinel_bands()
    
    print("\n" + "="*60)
    print("  COLLECTION COMPLETE!")
    print("="*60)
    print(f"\n  Project data directory: {os.path.join(PROJECT, 'data')}")
    print(f"  Models directory: {os.path.join(PROJECT, 'models')}")


if __name__ == "__main__":
    main()
