import matplotlib.pyplot as plt
import cv2
import numpy as np
from tkinter import filedialog

def proses_citra():
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)

    x, y, w, h = 100, 100, 300, 300 
    cropped_image = image[y:y+h, x:x+w]

    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    segmented_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("Citra Cropping")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title("Segmentasi Warna")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([segmented_image], [i], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title("Histogram RGB")

    plt.tight_layout()
    plt.show()

proses_citra()
