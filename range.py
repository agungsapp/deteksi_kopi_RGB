from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Paths
model_path = 'coffee_model.h5'
test_dir = 'dataset/test'

# Load model
model = load_model(model_path)

# Prepare data generator
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

color_data = {}

test_generator.reset()

for i in range(len(test_generator)):
    img, label = next(test_generator)
    prediction = model.predict(img)
    predicted_class = list(test_generator.class_indices.keys())[np.argmax(prediction[0])]

    img_255 = (img[0] * 255).astype(np.uint8)

    if predicted_class not in color_data:
        color_data[predicted_class] = {
            'min': np.min(img_255, axis=(0, 1)),
            'max': np.max(img_255, axis=(0, 1)),
            'sum': np.sum(img_255, axis=(0, 1)),
            'count': 1
        }
    else:
        color_data[predicted_class]['min'] = np.minimum(color_data[predicted_class]['min'], np.min(img_255, axis=(0, 1)))
        color_data[predicted_class]['max'] = np.maximum(color_data[predicted_class]['max'], np.max(img_255, axis=(0, 1)))
        color_data[predicted_class]['sum'] += np.sum(img_255, axis=(0, 1))
        color_data[predicted_class]['count'] += 1

for label, data in color_data.items():
    print(f"Label: {label}")
    print(f"  Min RGB: {data['min']}")
    print(f"  Max RGB: {data['max']}")
    print(f"  Average RGB: {data['sum'] / (data['count'] * 224 * 224)}")
    print()