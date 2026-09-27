"""
Team Brain Wave - Sentinel-2 Search via correct STAC API
Uses: https://stac.dataspace.copernicus.eu/v1/search
"""
import urllib.request
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sentinel2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIONS = {
    "punjab": [74.5, 30.0, 76.5, 32.0],
    "assam": [89.5, 25.5, 92.0, 27.0],
    "delhi": [76.8, 28.3, 77.5, 28.9],
    "himalaya": [78.0, 30.0, 79.5, 31.0],
    "gujarat": [69.0, 22.5, 71.0, 24.0],
}

def search_stac(bbox, date_range, cloud_max=30, limit=5):
    url = "https://stac.dataspace.copernicus.eu/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": date_range,
        "query": {"eo:cloud_cover": {"lt": cloud_max}},
        "limit": limit,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}]
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        'Content-Type': 'application/json',
        'Accept': 'application/geo+json'
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result.get('features', [])
    except Exception as e:
        print(f"  Error: {e}")
        return []

def main():
    print("=" * 60)
    print("  Team Brain Wave - Sentinel-2 L2A Search")
    print("=" * 60)
    
    all_products = []
    
    for name, bbox in REGIONS.items():
        print(f"\n--- Region: {name.upper()} (bbox: {bbox}) ---")
        
        # Search for clear winter images
        results = search_stac(bbox, "2025-01-01T00:00:00Z/2025-02-28T23:59:59Z", cloud_max=20, limit=3)
        if results:
            print(f"  Winter (clear): {len(results)} products found")
            for r in results:
                p = r.get('properties', {})
                cloud = p.get('eo:cloud_cover', 'N/A')
                dt = p.get('datetime', 'N/A')[:10]
                pid = r.get('id', 'unknown')
                print(f"    - {pid[:50]}... | cloud={cloud}% | date={dt}")
                
                # Check for quicklook/thumbnail
                assets = r.get('assets', {})
                for akey in ['quicklook', 'thumbnail', 'preview']:
                    if akey in assets:
                        ql_url = assets[akey].get('href', '')
                        if ql_url:
                            print(f"      Quicklook: {ql_url[:80]}...")
                            break
                
                all_products.append({
                    "id": pid,
                    "region": name,
                    "season": "winter_clear",
                    "cloud_cover": cloud,
                    "date": dt,
                    "assets": {k: v.get('href', '') for k, v in assets.items() if 'href' in v}
                })
        else:
            print(f"  Winter: No results")
        
        # Search for monsoon images (cloudy)
        results = search_stac(bbox, "2025-07-01T00:00:00Z/2025-08-31T23:59:59Z", cloud_max=90, limit=3)
        if results:
            print(f"  Monsoon (cloudy): {len(results)} products found")
            for r in results:
                p = r.get('properties', {})
                cloud = p.get('eo:cloud_cover', 'N/A')
                dt = p.get('datetime', 'N/A')[:10]
                pid = r.get('id', 'unknown')
                print(f"    - {pid[:50]}... | cloud={cloud}% | date={dt}")
                all_products.append({
                    "id": pid,
                    "region": name,
                    "season": "monsoon_cloudy",
                    "cloud_cover": cloud,
                    "date": dt
                })
        else:
            print(f"  Monsoon: No results")
    
    # Save catalog
    catalog_file = os.path.join(OUTPUT_DIR, "sentinel2_catalog.json")
    with open(catalog_file, 'w', encoding='utf-8') as f:
        json.dump(all_products, f, indent=2)
    print(f"\nCatalog saved: {catalog_file} ({len(all_products)} products)")


if __name__ == "__main__":
    main()
