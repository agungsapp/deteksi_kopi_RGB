from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

model_path = 'coffee_model.h5'
test_dir = 'dataset/test'  


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

for i in range(5):
    img, label = next(test_generator) 
    prediction = model.predict(img)
    plt.imshow(img[0])
    plt.title(f"True Label: {list(test_generator.class_indices.keys())[np.argmax(label[0])]} - Prediction: {list(test_generator.class_indices.keys())[np.argmax(prediction[0])]}")
    plt.show()
