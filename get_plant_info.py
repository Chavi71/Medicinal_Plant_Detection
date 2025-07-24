import pandas as pd

def get_plant_info(plant_name):
    # Load the corrected CSV file
    data = pd.read_csv(r"C:\Users\Amulya H G\OneDrive\Desktop\PC1 Desktop data\BioBotanica\data\plant_info.csv", on_bad_lines='skip')

    # Normalize column names (remove spaces and make lowercase)
    data.columns = data.columns.str.strip().str.lower()

    # Search for the plant by botanical or common name
    plant_info = data[
        (data["botanical_name"].str.lower() == plant_name.lower()) |
        (data["common_name"].str.lower() == plant_name.lower())
    ]
    
    if not plant_info.empty:
        info = plant_info.iloc[0]  # Get the first match
        return {
            "description": info["description"],
            "uses": info["uses"],
            "benefits": info["benefits"],
            "how_to_grow": info["how_to_grow"]
        }
    else:
        return "No information available for this plant."
