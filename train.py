import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Data augmentation untuk training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Hanya rescaling untuk testing
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Menyiapkan generator untuk training set
train_generator = train_datagen.flow_from_directory(
    'D:/JOKI/PYTHON/KOPI/DATASET/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['A', 'B', 'C']  # Disesuaikan dengan nama folder baru
)

# Menyiapkan generator untuk testing set
test_generator = test_datagen.flow_from_directory(
    'D:/JOKI/PYTHON/KOPI/DATASET/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['A', 'B', 'C']  # Disesuaikan dengan nama folder baru
)

# Membuat model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output untuk 3 kelas: A, B, C
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Simpan model
model.save('coffee_model.h5')
