import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def is_brown(rgb):
    r, g, b = rgb
    return (r > g > b) and (r > 60) and (g > 30) and (b < 50)

def get_average_color(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.apply_along_axis(is_brown, 2, image_rgb)
    res = image_rgb * mask[:,:,np.newaxis]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = res[y:y+h, x:x+w]

        flattened = cropped.reshape(-1, 3)
        mask = np.apply_along_axis(is_brown, 1, flattened)
        filtered = flattened[mask]

        if filtered.size > 0:
            avg_color = np.mean(filtered, axis=0)
            return avg_color.astype(int)
    
    return None

def process_dataset(dataset_path, grade, limit_images):
    grade_path = os.path.join(dataset_path, grade)
    if not os.path.exists(grade_path):
        print(f"Folder tidak ditemukan: {grade_path}")
        return []

    results = []
    image_files = [f for f in os.listdir(grade_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, image_file in enumerate(tqdm(image_files[:limit_images], desc=f"Memproses {grade}")): 
        image_path = os.path.join(grade_path, image_file)
        avg_color = get_average_color(image_path)
        
        if avg_color is not None:
            results.append({
                    'No': i+1,
                    'Nama Objek/Gambar': os.path.splitext(image_file)[0], 
                    'Red (merah)': avg_color[0],
                    'Green (hijau)': avg_color[1],
                    'Blue (biru)': avg_color[2]
            })

    return results

def save_to_excel(results, output_path):
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Hasil telah disimpan ke: {output_path}")

if __name__ == "__main__":
    dataset_path = r"D:\JOKI\PYTHON\KOPI\DATASET\train"
    grade = "A"  
    output_path = f"excel/train/hasil_rata_rata_{grade}.xlsx"
    limit_images = 50

    results = process_dataset(dataset_path, grade, limit_images)
    save_to_excel(results, output_path)