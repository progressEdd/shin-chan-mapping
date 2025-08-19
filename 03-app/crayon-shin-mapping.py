import os
import re
from dotenv import load_dotenv
from pathlib import Path
from vector_search import vector_search
from typing import List, Dict, Any
import copy
import pandas as pd

from pathlib import Path

try:
    start = Path(__file__).resolve()
except NameError:
    start = Path.cwd()

supporting_files = next(p / "00-supporting-files" for p in start.parents if (p / "00-supporting-files").exists())

data_path = supporting_files / "data" 
wikipedia_path = data_path / 'wikipedia_episodes.json'
german_path = data_path / "wunschliste_episodes.json"

embedding_model = "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:latest"
ollama_llm = "gemma3:27b-it-q4_K_M"

import ollama
from openai import OpenAI
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

import json
with wikipedia_path.open("r") as f:
    wikipedia_data = json.load(f)

with german_path.open("r") as f:
    wunschliste_data = json.load(f)

def clean_title(title: str) -> str:
    # Normalize curly quotes
    title = title.replace("“", '"').replace("”", '"').replace("’", "'")

    # Replace (Japanese: …) → keep inside, add a leading space
    title = re.sub(r"\(Japanese:\s*([^)]+)\)", r" \1", title, flags=re.IGNORECASE)

    # Replace (Transliteration: …) → keep inside, add a leading space
    title = re.sub(r"\(Transliteration:\s*([^)]+)\)", r" \1", title, flags=re.IGNORECASE)

    # Handle inline Transliteration/Japanese labels
    title = re.sub(r"Transliteration:\s*", " ", title, flags=re.IGNORECASE)
    title = re.sub(r"Japanese:\s*", " ", title, flags=re.IGNORECASE)

    # Remove stray quotes
    title = title.replace('"', "").replace("'", "")

    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    
    return title

def flatten_wikipedia(wikipedia_data):
    wiki_flat = []
    for year, episodes in wikipedia_data.items():
        for ep in episodes:
            for title in ep["titles"]:
                wiki_flat .append({
                "year": year,
                "episode": ep["episode"],
                "title": clean_title(title),
                "description": ep.get("description", "")
            })
    return wiki_flat

def flatten_wunschliste(wunschliste_data):
    wuns_flat = []
    for season, episodes in wunschliste_data.items():
        for ep in episodes:
            for title in ep["titles"]:
                wuns_flat.append({
                    "season": season.replace("season ", ""),
                    "episode": ep["episode"],
                    "title": title
                })                    
    return wuns_flat

def generate_embedding(data, embedding_model):
    for row in data:
        row["title_embedding"] = ollama.embed(model=embedding_model,input=row["title"]).embeddings
    return data

def export_embeddings(data, data_path, filename):
    embedding_export = data_path / filename
    with open(embedding_export, "w", encoding="utf-8") as f:
        json.dump(wiki_flat, f, indent=2)

def create_title_list(matches):
    titles = []
    for entry in matches:
        # print(entry)
        titles.append(entry["title"])
    return titles # list of titles

def remove_by_embedded_title(episodes: List[Dict[str, Any]], title: str) -> List[Dict[str, Any]]:
    """
    Return a new list with any entries whose 'title' exactly equals `title` removed.
    Does not modify the original list.
    """
    return [ep for ep in episodes if ep.get("title") != title]



def main():
    # generate wikipedia data, uncomment if you want to generate wikipedia
    wiki_flat = flatten_wikipedia(wikipedia_data)
    wiki_flat_embedded = generate_embedding(wiki_flat, embedding_model)
    # export_embeddings(wuns_flat_embedded, "flattened_wikipedia.json") # uncomment if you want to export embeddings

    wuns_flat = flatten_wunschliste(wunschliste_data)
    wuns_flat_embedded = generate_embedding(wuns_flat, embedding_model)
    # export_embeddings(wiki_flat_embedded, "flattened_wunschliste.json")
    


    for idx, episode in enumerate(wuns_flat_embedded): # replace wuns_flat_embedded with wiki_flat_embedded if you want to map english to to german titles, make sure to also update the title language and retrieved_title_language
        title_to_map = episode["title"]
        title_language =  "German"

        
        matches = vector_search(
            wiki_flat, # replace with wuns if searching on german titles
            text=title_to_map,
            model=embedding_model,
            top_k=30,
            return_fields=["year", "episode", "title", "description"] 
            
        )

        retrieved_titles = get_title_list(matches)
        retrieved_title_language = "English/Japanese"


        llm_matches = llm_matching(title_to_map, title_language,
                                   retrieved_titles, retrieved_title_language, 
                                   llm_model=ollama_llm)
        
        mapped_title = llm_matches["mapped_title"]

        # ✅ Attach metadata for the chosen mapped title
        mapped_meta = next((m for m in matches if m["title"] == mapped_title), None)

        episode["mapped_title"] = mapped_title
        episode["mapping_reason"] = llm_matches["reason"]
        episode["mapping_rejection_reason"] = llm_matches["negatives"]
        episode["search_results"] = matches
        episode["mapped_meta"] = mapped_meta   # ← here is your meta info

        # Remove the chosen title from vector index to avoid re-mapping
        vector_index = remove_by_embedded_title(vector_index, mapped_title)
    
    removed_embeddings = copy.deepcopy(wuns_flat)
    for episode in removed_embeddings:
        episode.pop("title_embedding")

    embedding_export = data_path / "mapped_wuns.json"
    with open(embedding_export, "w", encoding="utf-8") as f:
        json.dump(wuns_flat, f, indent=2)

    markdown_path = data_path / "mapped_episodes_table.md"
    markdown_df = pd.DataFrame(wuns_flat)
    markdown_df.drop(columns="search_results", inplace=True)
    markdown_str = markdown_df.to_markdown(index=False)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_str)

if __name__ == "__main__":
    main()