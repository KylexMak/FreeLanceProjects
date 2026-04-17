import requests

def get_stats():
    sets_url = 'https://api.tcgdex.net/v2/en/sets'
    sets = requests.get(sets_url).json()
    pocket_sets = [s['id'] for s in sets if s['id'].startswith('A') or s['id'].startswith('B')]
    
    total_cards = 0
    rare_count = 0
    
    for s_id in pocket_sets:
        cards = requests.get(f'{sets_url}/{s_id}').json().get('cards', [])
        total_cards += len(cards)
        for c in cards:
            rarity = c.get('rarity', '')
            if rarity and rarity.lower() != 'common':
                rare_count += 1
                
    print(f"Total Unique Cards: {total_cards}")
    print(f"Total Rare/Higher Cards: {rare_count}")

if __name__ == "__main__":
    get_stats()
