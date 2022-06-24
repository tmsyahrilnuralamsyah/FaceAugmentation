from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping


def optimized_data(train_data_path, BS, train=True):
    if train:
        print("Training Data :")
    else:
        print("Validation Data :")
        
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    train_dataset = train_datagen.flow_from_directory(
        directory=train_data_path,
        target_size=(224, 224),
        class_mode="categorical",
        batch_size=BS
    )
    train_class_names = train_dataset.class_indices
    return train_dataset, train_class_names

def resnet50(output_class):
    return ResNet50(classes=output_class, weights=None, input_shape=(224, 224, 3))

def mobilenetv3small(output_class):
    return MobileNetV3Small(classes=output_class, weights=None, input_shape=(224, 224, 3))

def mobilenetv3large(output_class):
    return MobileNetV3Large(classes=output_class, weights=None, input_shape=(224, 224, 3))

def my_callbacks(MODEL_NAME, label_size, image_per_label, EPOCHS, BS):
    model_callbacks = [
        # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        CSVLogger(
            filename=f"Model/{MODEL_NAME}/history_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}.csv", separator=",", append=False),
        ModelCheckpoint(
            filepath=f"Model/{MODEL_NAME}/model_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}_val_loss.h5", monitor="val_loss", save_best_only=True),
        ModelCheckpoint(
            filepath=f"Model/{MODEL_NAME}/model_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}_val_accuracy.h5", monitor="val_accuracy", save_best_only=True)
    ]
    return model_callbacks

def load_my_model(MODEL_NAME, label_size, image_per_label, EPOCHS, BS):
    model = load_model(
        f"Model/{MODEL_NAME}/model_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}_val_accuracy.h5")
    return model
