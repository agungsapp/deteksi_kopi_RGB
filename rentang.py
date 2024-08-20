import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from tkinter import Tk, Label
from PIL import Image, ImageTk

def is_brown(rgb):
    r, g, b = rgb
    return (r > g > b) and (r > 60) and (g > 30) and (b < 50)

def get_coffee_color_range(image_path, show_result=False):
   
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.apply_along_axis(is_brown, 2, image_rgb)
    res = image_rgb * mask[:,:,np.newaxis]

    # test kontur dan crop area
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped = res[y:y+h, x:x+w]

        # filter warna coklat
        flattened = cropped.reshape(-1, 3)
        mask = np.apply_along_axis(is_brown, 1, flattened)
        filtered = flattened[mask]

        if filtered.size > 0:
            # Hitung min dan max RGB
            min_rgb = np.percentile(filtered, 5, axis=0)
            max_rgb = np.percentile(filtered, 95, axis=0)

            if show_result:
                # yoi crop area 
                root = Tk()
                root.title("Crop Area")
                img = Image.fromarray(cropped)
                img = ImageTk.PhotoImage(img)
                panel = Label(root, image=img)
                panel.pack(side="bottom", fill="both", expand="yes")
                root.mainloop()

                #  visualisasi oke
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(image_rgb)
                plt.title("Original Image")
                plt.subplot(132)
                plt.imshow(cropped)
                plt.title("Cropped Image")
                plt.subplot(133)
                plt.imshow(filtered.reshape(-1, 1, 3))
                plt.title("Filtered Colors")
                plt.show()

            return min_rgb, max_rgb
    
    return None, None

def process_dataset(dataset_path):
    grades = ['Dark', 'Light', 'Medium']
    results = {grade: {'min': [], 'max': []} for grade in grades}

    for grade in grades:
        grade_path = os.path.join(dataset_path, grade)
        if not os.path.exists(grade_path):
            print(f"Folder tidak ditemukan: {grade_path}")
            continue

        image_files = [f for f in os.listdir(grade_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc=f"Memproses {grade}"):
            image_path = os.path.join(grade_path, image_file)
            min_rgb, max_rgb = get_coffee_color_range(image_path, show_result=(image_file == image_files[0]))
            
            if min_rgb is not None and max_rgb is not None:
                results[grade]['min'].append(min_rgb)
                results[grade]['max'].append(max_rgb)

    # Hitung rata-rata rentang untuk setiap grade
    for grade in grades:
        if results[grade]['min'] and results[grade]['max']:
            avg_min = np.mean(results[grade]['min'], axis=0)
            avg_max = np.mean(results[grade]['max'], axis=0)
            print(f"\nRentang warna untuk {grade}:")
            print(f"Min RGB: {avg_min.astype(int)}")
            print(f"Max RGB: {avg_max.astype(int)}")
        else:
            print(f"\nTidak ada data valid untuk {grade}")

if __name__ == "__main__":
    dataset_path = r"./DATASET\test"  # Sesuaikan dengan path dataset Anda
    process_dataset(dataset_path)