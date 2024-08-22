from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

model_path = 'coffee_model.h5'  
test_dir = 'dataset/test' 

# Load model
model = load_model(model_path)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

color_ranges = {}

for i in range(len(test_generator)):
    img, label = test_generator[i]
    img_rgb = img[0] 

    prediction = model.predict(np.expand_dims(img_rgb, axis=0))
    predicted_class = np.argmax(prediction, axis=1)[0]

    gray = cv2.cvtColor((img_rgb * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    mask = np.zeros_like(gray, dtype=np.uint8)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    try:
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)

            if gray.dtype == 'uint8' and mask.dtype == 'uint8':
                min_val, max_val, _, _ = cv2.minMaxLoc((img_rgb * 255).astype('uint8'), mask=mask)

                if predicted_class not in color_ranges:
                    color_ranges[predicted_class] = {'min': min_val, 'max': max_val}
                else:
                    color_ranges[predicted_class]['min'] = np.minimum(color_ranges[predicted_class]['min'], min_val)
                    color_ranges[predicted_class]['max'] = np.maximum(color_ranges[predicted_class]['max'], max_val)
            else:
                print(f"Tipe data mask atau gambar tidak sesuai pada gambar ke-{i}")
        else:
            print(f"Tidak ada kontur yang ditemukan pada gambar ke-{i}")

    except Exception as e:
        print(f"Error saat memproses gambar ke-{i}: {e}")

for label, ranges in color_ranges.items():
    print(f"Label: {label}")
    print(f"  Min RGB: {ranges['min']}")
    print(f"  Max RGB: {ranges['max']}")
