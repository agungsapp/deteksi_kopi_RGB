from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np

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
    global panelA, result_label

    if hasattr(root, 'path'):
        try:
            image = cv2.imread(root.path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

            roi_image = cv2.resize(image_rgb, (224, 224)) 
            roi_image = img_to_array(roi_image)
            roi_image = np.expand_dims(roi_image, axis=0)
            roi_image /= 255.0

            prediction = model.predict(roi_image)
            print("Prediction:", prediction) 
            class_labels = ['A', 'C', 'B']
            predicted_class = class_labels[np.argmax(prediction)]

            result_label.config(text=f"Hasil Prediksi: {predicted_class}")

        except Exception as e:
            print("Error during prediction:", e)



button = Button(root, text = "PROSES CITRA",bg="blue",fg="white",
                 command = proses_citra)
button.pack(side="right", fill="y",ipadx="10",ipady="5",padx="10", pady="5")
label1.pack()

root.mainloop()
