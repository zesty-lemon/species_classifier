import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor

# 1. Load the labels and image URLs from the unzipped GBIF files
print("Loading data...")
occ = pd.read_csv('./gbif_data/occurrence.txt', sep='\t', low_memory=False, usecols=['gbifID', 'species'])
media = pd.read_csv('./gbif_data/multimedia.txt', sep='\t', low_memory=False, usecols=['gbifID', 'identifier', 'format'])

# 2. Merge them together and filter for images only
df = pd.merge(occ, media, on='gbifID')
df = df[df['format'].str.contains('image', na=False, case=False)]

# Drop records that don't have a defined species label or URL
df = df.dropna(subset=['species', 'identifier'])
print(f"Found {len(df)} images to process.")

# 3. Define the download logic
def download_image(row):
    # Create a clean folder name for the species (e.g., "Pitta_nympha")
    species_name = str(row['species']).replace(" ", "_")
    folder = f"my_dataset/{species_name}"
    os.makedirs(folder, exist_ok=True)
    
    file_path = os.path.join(folder, f"{row['gbifID']}.jpg")
    
    # Skip if we already downloaded it (allows you to pause/resume the script safely)
    if os.path.exists(file_path): 
        return 
    
    try:
        r = requests.get(row['identifier'], timeout=5)
        if r.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(r.content)
    except Exception:
        pass # Ignore dead links quietly

# 4. Download concurrently for speed (10 simultaneous downloads)
print("Starting fast download...")
with ThreadPoolExecutor(max_workers=10) as executor:
    # Convert dataframe to dictionaries for fast iteration
    executor.map(download_image, df.to_dict('records'))

print("Dataset extraction complete!")