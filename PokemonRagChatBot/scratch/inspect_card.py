import requests
import json

def inspect_card(card_id):
    url = f"https://api.tcgdex.net/v2/en/cards/{card_id}"
    resp = requests.get(url)
    card = resp.json()
    print(json.dumps(card, indent=2))

if __name__ == "__main__":
    inspect_card("A1-157")
