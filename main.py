from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('coffee_model.h5')

root = Tk() 
root.title("Deteksi Kualitas Biji Kopi") 
root.geometry("1100x600")

label1 = Label(root, text = "SELAMAT DATANG\n" 
"DI GRAPHICAL USER INTERFACE\n"
"DETEKSI KUALITAS BIJI KOPI\n",
font=("Arial",25))
label1.pack(side="top", fill="both",padx="10",pady="10")

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack()

panelA = None

def select_image():
    global panelA

    root.path = filedialog.askopenfilename()
    if len(root.path) > 0:
        image = cv2.imread(root.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((500, 500))
        image = ImageTk.PhotoImage(image) 
        if panelA is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(fill="x", padx=5, pady=5)
        else:
            panelA.configure(image=image)
            panelA.image = image

def Mulai():
    button2 = Button(root, text="PILIH GAMBAR KOPI", bg="green",fg="white",command=select_image)
    button2.pack(side="left", fill="y",ipadx="10",ipady="5",padx="10", pady="5")

button1=Button(root,text="MULAI", bg="red",fg="white",command=Mulai)
button1.pack(side="left",fill="y", ipadx="10", ipady="5",padx="10", pady="5")
print("_____MULAI______")

def proses_citra():
    if hasattr(root, 'path'):
        try:
            image1 = cv2.imread(root.path)
            rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            
            # Thresholding untuk segmentasi warna
            lower = np.array([0, 0, 0])
            upper = np.array([179, 172, 165])
            mask = cv2.inRange(rgb, lower, upper)
            res = cv2.bitwise_and(rgb, rgb, mask=mask)

            # Deteksi kontur dan crop area yang relevan
            cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cont, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped = res[y:y+h, x:x+w]
            
            # Hitung rata-rata RGB
            avg_color_per_row = np.average(cropped, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            R, G, B = avg_color
            R, G, B = round(R, 2), round(G, 2), round(B, 2)
            
            # Konversi ke HSV
            hsv_image = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
            H, S, V = cv2.mean(hsv_image)[:3]
            H = round(H * 2, 1)  # Scaling H to 0-360 range
            S = round(S / 255 * 100, 1)
            V = round(V / 255 * 100, 1)

            # Prediksi menggunakan model
            roi_image = cv2.resize(rgb, (224, 224)) 
            roi_image = img_to_array(roi_image)
            roi_image = np.expand_dims(roi_image, axis=0)
            roi_image /= 255.0

            prediction = model.predict(roi_image)
            class_labels = ['A', 'C', 'B']
            predicted_class = class_labels[np.argmax(prediction)]

            # Create new window for results
            top1 = Toplevel(root)
            top1.title("Proses Citra")
            top1.geometry("1100x800")

            fig, axes = plt.subplots(2, 3, figsize=(10, 8))
            
            # Gambar awal
            axes[0, 0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Citra Awal')
            axes[0, 0].axis('off')

            # Gambar cropping
            axes[0, 1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Citra Cropping')
            axes[0, 1].axis('off')

            # Nilai rata-rata RGB
            axes[0, 2].text(0.2, 0.7, "Nilai Rata-Rata RGB", fontsize=12)
            axes[0, 2].text(0.2, 0.5, f"Merah = {R}", fontsize=12)
            axes[0, 2].text(0.2, 0.35, f"Hijau = {G}", fontsize=12)
            axes[0, 2].text(0.2, 0.2, f"Biru = {B}", fontsize=12)
            axes[0, 2].axis('off')

            # Citra HSV
            axes[1, 0].imshow(hsv_image)
            axes[1, 0].set_title('Citra HSV')
            axes[1, 0].axis('off')

            # Nilai HSV
            axes[1, 1].text(0.2, 0.8, f"H = {H}°", fontsize=12)
            axes[1, 1].text(0.2, 0.5, f"S = {S}%", fontsize=12)
            axes[1, 1].text(0.2, 0.2, f"V = {V}%", fontsize=12)
            axes[1, 1].axis('off')

            # Kualitas
            axes[1, 2].text(0.2, 0.5, f"Kualitas: {predicted_class}", fontsize=15, color='red')
            axes[1, 2].axis('off')

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=top1)
            canvas.draw()
            canvas.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=1)

            result_label.config(text=f"Hasil Prediksi: {predicted_class}")

        except Exception as e:
            print("Error during prediction:", e)

button = Button(root, text = "PROSES CITRA",bg="blue",fg="white", command = proses_citra)
button.pack(side="right", fill="y",ipadx="10",ipady="5",padx="10", pady="5")
label1.pack()

root.mainloop()