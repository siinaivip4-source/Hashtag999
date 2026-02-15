# app_upgraded.py

import os
import pandas as pd
import sqlite3

# Dummy function for ViT-B/32 model
def use_vit_b32_model(image_path):
    # Placeholder for model processing
    return ["#emotion1", "#gender1", "#hashtag1", "#hashtag2"]

def process_batch_images(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            hashtags = use_vit_b32_model(image_path)
            results.append({"image": filename, "hashtags": hashtags})
    return results

def export_to_excel(results, output_path):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)

def export_to_sql(results, db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS hashtags (image TEXT, hashtags TEXT)')
    for result in results:
        c.execute('INSERT INTO hashtags (image, hashtags) VALUES (?, ?)', (result['image'], ', '.join(result['hashtags'])))
    conn.commit()
    conn.close()

def main(folder_path, excel_output, sql_db_name):
    results = process_batch_images(folder_path)
    export_to_excel(results, excel_output)
    export_to_sql(results, sql_db_name)

if __name__ == "__main__":
    # Example usage
    main("path/to/image/folder", "output.xlsx", "hashtags.db")
