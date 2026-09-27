"""
Team Brain Wave — ISRO BAH 2026
Sentinel-2 Data Search & Preview Downloader
Uses Copernicus Data Space STAC API (free, no auth needed for search)
"""

import urllib.request
import json
import os
import sys
from datetime import datetime

# =====================================================
# CONFIG — Indian Regions of Interest for Cloud Removal
# =====================================================
REGIONS = {
    "punjab_agriculture": {
        "bbox": [74.5, 30.0, 76.5, 32.0],
        "description": "Punjab agricultural belt — rice/wheat monitoring"
    },
    "assam_flood": {
        "bbox": [89.5, 25.5, 92.0, 27.0],
        "description": "Assam Brahmaputra floodplain — flood response"
    },
    "delhi_ncr_urban": {
        "bbox": [76.8, 28.3, 77.5, 28.9],
        "description": "Delhi NCR — urban sprawl monitoring"
    },
    "himalayan_foothill": {
        "bbox": [78.0, 30.0, 79.5, 31.0],
        "description": "Uttarakhand Himalayan foothills — snow/cloud distinction"
    },
    "gujarat_kutch": {
        "bbox": [69.0, 22.5, 71.0, 24.0],
        "description": "Gujarat Kutch — arid/semi-arid, NHAI green corridor"
    }
}

# Date ranges for monsoon (high cloud) vs clear season comparison
DATE_RANGES = {
    "monsoon_cloudy": ("2025-07-01", "2025-08-31"),
    "clear_winter": ("2025-01-01", "2025-02-28"),
    "pre_monsoon": ("2025-04-01", "2025-05-31")
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "sentinel2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def search_sentinel2(bbox, start_date, end_date, max_cloud=80, limit=5):
    """
    Search Copernicus Data Space STAC API for Sentinel-2 L2A products.
    Returns list of product metadata dicts.
    """
    stac_url = "https://catalogue.dataspace.copernicus.eu/stac/search"
    
    payload = {
        "collections": ["SENTINEL-2"],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "filter": f"eo:cloud_cover < {max_cloud}",
        "filter-lang": "cql2-text",
        "sortby": [{"field": "properties.datetime", "direction": "desc"}]
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        stac_url,
        data=data,
        headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('features', [])
    except Exception as e:
        print(f"  [WARN] STAC search failed: {e}")
        return []


def search_sentinel2_odata(bbox, start_date, end_date, max_cloud=80, limit=5):
    """
    Fallback: Search using OData API for Sentinel-2 products.
    """
    west, south, east, north = bbox
    footprint = f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))')"
    
    url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$filter=Collection/Name eq 'SENTINEL-2'"
        f" and ContentDate/Start gt {start_date}T00:00:00.000Z"
        f" and ContentDate/Start lt {end_date}T23:59:59.000Z"
        f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud})"
        f" and {footprint}"
        f"&$orderby=ContentDate/Start desc"
        f"&$top={limit}"
    )
    
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('value', [])
    except Exception as e:
        print(f"  [WARN] OData search failed: {e}")
        return []


def download_quicklook(product_id, product_name, save_dir):
    """
    Downloads the quicklook/thumbnail for a Sentinel-2 product.
    """
    # Copernicus quicklook URL pattern
    ql_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/Nodes({product_name})/Nodes(GRANULE)/Nodes"
    
    # Alternative: use the zipper service for quicklook
    ql_url2 = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/Quicklook"
    
    filename = os.path.join(save_dir, f"{product_name}_quicklook.jpg")
    if os.path.exists(filename):
        print(f"  [SKIP] Already exists: {os.path.basename(filename)}")
        return filename
    
    try:
        urllib.request.urlretrieve(ql_url2, filename)
        size = os.path.getsize(filename)
        if size > 1000:
            print(f"  [OK] Downloaded quicklook ({size/1024:.1f} KB): {os.path.basename(filename)}")
            return filename
        else:
            os.remove(filename)
            print(f"  [WARN] Quicklook too small, likely auth required")
            return None
    except Exception as e:
        print(f"  [WARN] Quicklook download failed: {e}")
        return None


def save_search_results(results, region_name, season_name, save_dir):
    """Save search results as JSON for reference."""
    filename = os.path.join(save_dir, f"search_{region_name}_{season_name}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [SAVED] {len(results)} results → {os.path.basename(filename)}")
    return filename


def main():
    print("=" * 70)
    print("  Team Brain Wave — Sentinel-2 Data Discovery & Collection")
    print("  ISRO Bharatiya Antariksh Hackathon (BAH) 2026")
    print("=" * 70)
    print()
    
    all_results = {}
    
    for region_name, region_info in REGIONS.items():
        print(f"\n🛰️  Region: {region_name}")
        print(f"   {region_info['description']}")
        print(f"   Bbox: {region_info['bbox']}")
        
        region_dir = os.path.join(OUTPUT_DIR, region_name)
        os.makedirs(region_dir, exist_ok=True)
        
        for season_name, (start, end) in DATE_RANGES.items():
            cloud_max = 90 if season_name == "monsoon_cloudy" else 30
            print(f"\n   📅 Season: {season_name} ({start} to {end}), max cloud: {cloud_max}%")
            
            # Try STAC API first
            results = search_sentinel2(
                bbox=region_info['bbox'],
                start_date=start,
                end_date=end,
                max_cloud=cloud_max,
                limit=3
            )
            
            if not results:
                # Fallback to OData
                print("   Trying OData API fallback...")
                odata_results = search_sentinel2_odata(
                    bbox=region_info['bbox'],
                    start_date=start,
                    end_date=end,
                    max_cloud=cloud_max,
                    limit=3
                )
                if odata_results:
                    results = odata_results
                    for r in results:
                        pid = r.get('Id', 'unknown')
                        pname = r.get('Name', 'unknown')
                        cloud = "N/A"
                        attrs = r.get('Attributes', [])
                        for attr in attrs:
                            if attr.get('Name') == 'cloudCover':
                                cloud = f"{attr.get('Value', 'N/A'):.1f}%"
                        print(f"     ✓ {pname[:60]}... | Cloud: {cloud}")
                    
                    save_search_results(odata_results, region_name, season_name, region_dir)
                else:
                    print("   No results found.")
            else:
                for r in results:
                    props = r.get('properties', {})
                    title = props.get('title', r.get('id', 'unknown'))
                    cloud = props.get('eo:cloud_cover', 'N/A')
                    date = props.get('datetime', 'N/A')
                    print(f"     ✓ {title[:60]}... | Cloud: {cloud}% | Date: {date[:10]}")
                
                save_search_results(results, region_name, season_name, region_dir)
            
            key = f"{region_name}_{season_name}"
            all_results[key] = len(results) if results else 0
    
    # Summary
    print("\n" + "=" * 70)
    print("  SEARCH SUMMARY")
    print("=" * 70)
    total = 0
    for key, count in all_results.items():
        print(f"  {key}: {count} products found")
        total += count
    print(f"\n  TOTAL: {total} Sentinel-2 products discovered across all regions/seasons")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
