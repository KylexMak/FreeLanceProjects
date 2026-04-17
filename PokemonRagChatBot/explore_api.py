import requests
import json

def explore_api():
    base_url = "https://api.tcgdex.net/v2/en"
    
    print("Fetching series...")
    series_resp = requests.get(f"{base_url}/series")
    series_list = series_resp.json()
    
    pocket_series = None
    for s in series_list:
        if "pocket" in s.get("name", "").lower():
            pocket_series = s
            break
            
    if not pocket_series:
        print("Pocket series not found in main series list. Searching specifically...")
        for s in series_list:
            print(f"Found series: {s.get('name')} (ID: {s.get('id')})")
    else:
        print(f"Found Pocket Series: {pocket_series['name']} (ID: {pocket_series['id']})")
        
        print("\nFetching sets for this series...")
        sets_resp = requests.get(f"{base_url}/series/{pocket_series['id']}")
        sets_data = sets_resp.json()
        
        for s in sets_data.get("sets", []):
            print(f"Set: {s['name']} (ID: {s['id']})")

if __name__ == "__main__":
    explore_api()
