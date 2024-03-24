import os
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def build_model():
    inputs = Input(shape=(64, 64, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(train_generator.class_indices), activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


# Set paths
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'Train/Train')
validation_dir = os.path.join(base_dir, 'Validation/Validation')
test_dir = os.path.join(base_dir, 'Test/Test')

# Create label map
label_map = {
    0: 'Healthy',
    1: 'Powdery',
    2: 'Rust',
    # 0 : "Apple Apple scab",
    # 1 :	"Apple Black rot",
    # 2 :	"Apple Cedar apple rust",
    # 3 :	"Apple healthy",
    # 4 :	"Bacterial leaf blight in rice leaf",
    # 5 :	"Blight in corn Leaf",
    # 6 :	"Blueberry healthy",
    # 7 :	"Brown spot in rice leaf",
    # 8 :	"Cercospora leaf spot",
    # 9 :"Cherry (including sour) Powdery mildew",
    # 10:	"Cherry (including_sour) healthy",
    # 11:	"Common Rust in corn Leaf",
    # 12:	"Corn (maize) healthy",
    # 13:	"Garlic",
    # 14:	"Grape Black rot",
    # 15:	"Grape Esca Black Measles",
    # 16:	"Grape Leaf blight Isariopsis Leaf Spot",
    # 17:	"Grape healthy",
    # 18:	"Gray Leaf Spot in corn Leaf",
    # 19:	"Leaf smut in rice leaf",
    # 20:	"Orange Haunglongbing Citrus greening",
    # 21:	"Peach healthy",
    # 22:	"Pepper bell Bacterial spot",
    # 23:	"Pepper bell healthy",
    # 24:	"Potato Early blight",
    # 25:	"Potato Late blight",
    # 26:	"Potato healthy",
    # 27:	"Raspberry healthy",
    # 28:	"Sogatella rice",
    # 29:	"Soybean healthy",
    # 30:	"Strawberry Leaf scorch",
    # 31:	"Strawberry healthy",
    # 32:	"Tomato Bacterial spot",
    # 33:	"Tomato Early blight",
    # 34:	"Tomato Late blight",
    # 35:	"Tomato Leaf Mold",
    # 36:	"Tomato Septoria leaf spot",
    # 37:	"Tomato Spider mites Two spotted spider mite",
    # 38:	"Tomato Target Spot",
    # 39:	"Tomato Tomato mosaic virus",
    # 40:	"Tomato healthy",
    # 41:	"algal leaf in tea",
    # 42:	"anthracnose in tea",
    # 43:	"bird eye spot in tea",
    # 44:	"brown blight in tea",
    # 45:	"cabbage looper",
    # 46:	"corn crop",
    # 47:	"ginger",
    # 48:	"healthy tea leaf",
    # 49:	"lemon canker",
    # 50:	"onion",
    # 51:	"potassium deficiency in plant",
    # 52:	"potato crop",
    # 53:	"potato hollow heart",
    # 54:	"red leaf spot in tea",
    # 55:	"tomato canker"
}

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.2, 1.0],
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    classes=list(label_map.values())
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    classes=list(label_map.values())
)

# Model creation
model = build_model()

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mc = ModelCheckpoint('leaf_classifier.keras', monitor='val_loss', verbose=1, save_best_only=True)

# Model training
epochs = 100 #here
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    # callbacks=[earlystop, mc]
)

# Save the entire model
model.save("leaf_classifier.keras")

# Create a test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    classes=list(label_map.values())
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')

# Get the predicted probabilities for the test data
test_preds = model.predict(test_generator)

# Print the predicted labels and actual labels for a few samples
num_samples = 2

for i in range(num_samples):
    actual_label = np.argmax(test_generator.classes[i])
    predicted_label = np.argmax(test_preds[i])

    actual_label_name = label_map[actual_label]
    predicted_label_name = label_map[predicted_label]

    print(f'Sample {i + 1}:')
    print(f'Actual Label: {actual_label} ({actual_label_name})')
    print(f'Predicted Label: {predicted_label} ({predicted_label_name})')
    print()

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"