import requests
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json

def fetch_pocket_cards():
    base_url = "https://api.tcgdex.net/v2/en"
    series_id = "tcgp"
    
    print(f"Fetching sets for series: {series_id}...", flush=True)
    sets_resp = requests.get(f"{base_url}/series/{series_id}")
    sets_data = sets_resp.json()
    
    all_cards = []
    seen_keys = set()
    
    for set_id in sets_data.get("sets", []):
        set_id = set_id["id"]
        print(f"Fetching cards for set: {set_id}...", flush=True)
        cards_resp = requests.get(f"{base_url}/sets/{set_id}")
        cards_list = cards_resp.json().get("cards", [])
        num_cards = len(cards_list)
        
        for i, card_summary in enumerate(cards_list):
            card_id = card_summary["id"]
            
            if card_id in seen_keys:
                continue
            
            if i % 25 == 0:
                print(f"  Fetching details for card {i+1}/{num_cards}...", flush=True)
            
            detail_resp = requests.get(f"{base_url}/cards/{card_id}")
            card = detail_resp.json()
            all_cards.append(card)
            seen_keys.add(card_id)
            
    return all_cards

def process_card_to_doc(card):
    """Converts a card JSON object into a LangChain Document with a descriptive text."""
    name = card.get("name", "Unknown")
    set_name = card.get("set", {}).get("name", "Unknown Set")
    rarity = card.get("rarity", "Unknown Rarity")
    category = card.get("category", "Unknown Category")
    
    text_parts = [f"Card Name: {name}", f"Set: {set_name}", f"Category: {category}", f"Rarity: {rarity}"]
    
    if "Mega" in name and "Meganium" not in name and "Yanmega" not in name:
        text_parts.append("Special Feature: This is a Mega Evolution forms card.")
    
    if category == "Pokemon":
        hp = card.get("hp", "N/A")
        types = ", ".join(card.get("types", []))
        stage = card.get("stage", "Basic")
        evolve_from = card.get("evolveFrom")
        
        text_parts.append(f"Stage: {stage}")
        if evolve_from:
            text_parts.append(f"Evolves From: {evolve_from}")
            
        text_parts.append(f"HP: {hp}")
        text_parts.append(f"Types: {types}")
        
        abilities = card.get("abilities", [])
        for ability in abilities:
            text_parts.append(f"Ability: {ability.get('name')} - {ability.get('effect')}")
            
        attacks = card.get("attacks", [])
        for attack in attacks:
            cost = ", ".join(attack.get("cost", []))
            damage = attack.get("damage", "0")
            effect = attack.get("effect", "")
            text_parts.append(f"Attack: {attack.get('name')} (Cost: {cost}, Damage: {damage}) - {effect}")
            
        retreat = card.get("retreat", 0)
        text_parts.append(f"Retreat Cost: {retreat}")
        
    elif category == "Trainer":
        trainer_type = card.get("trainerType", "Trainer")
        text_parts.append(f"Trainer Type: {trainer_type}")
        effect = card.get("effect", "N/A")
        text_parts.append(f"Effect: {effect}")
        
    elif category == "Energy":
        energy_type = card.get("energyType", "Basic")
        text_parts.append(f"Energy Type: {energy_type}")
        effect = card.get("effect", "Provides Energy.")
        text_parts.append(f"Effect: {effect}")
        
    description = "\n".join(text_parts)
    
    metadata = {
        "id": card.get("id"),
        "name": name,
        "set": set_name,
        "image": f"{card.get('image')}/high.webp" if card.get("image") else None
    }
    
    return Document(page_content=description, metadata=metadata)

def main():
    print("Starting data ingestion...")
    all_cards = fetch_pocket_cards()
    print(f"Fetched {len(all_cards)} unique cards. Initializing embeddings...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    docs = [process_card_to_doc(c) for c in all_cards]
    
    #TODO: Make this more dynamic
    overview_text = (
        "GLOBAL LIBRARY OVERVIEW: This library contains a comprehensive collection of Pokemon TCG Pocket cards. "
        "It includes exactly 15 sets: Genetic Apex, Mythical Island, Space-Time Smackdown, Triumphant Light, "
        "Shining Revelry, Celestial Guardians, Extradimensional Crisis, Eevee Grove, Wisdom of Sea and Sky, "
        "Secluded Springs, Mega Rising, Crimson Blaze, Fantastical Parade, Paldean Wonders, and Promos-A. "
        "The collection features a total of 18 unique Mega Evolution forms cards. "
        "Total cards in library: approx 2,480. "
        "RARITY BREAKDOWN (unique card names): "
        "One Diamond: 448 unique cards. "
        "Two Diamond: 425 unique cards. "
        "Three Diamond: 245 unique cards. "
        "Four Diamond: 116 unique cards. "
        "One Star: 5 unique cards. "
        "Three Star: 1 unique card. "
        "One Shiny: 122 unique cards. "
        "Two Shiny: 55 unique cards. "
        "Crown: 2 unique cards."
    )
    overview_doc = Document(page_content=overview_text, metadata={"name": "Global Library Overview", "image": ""})
    docs.append(overview_doc)
    
    vector_store = FAISS.from_documents(docs, embeddings)
    
    print("Saving FAISS index locally...")
    vector_store.save_local("faiss_index")
    print("Done!")

if __name__ == "__main__":
    main()
