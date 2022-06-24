import os
import time
import numpy as np
import tensorflow as tf
from research.evaluation import visualize_accuracy, visualize_loss, print_class_reports, print_conf_matrix
from research.researchmodel import optimized_data, my_callbacks, load_my_model, resnet50, mobilenetv3small, mobilenetv3large
from tensorflow.keras.optimizers import Adam


# Use GPU of Server
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Initiate Constant
BS = 20
EPOCHS = 150
MODEL_NAME = "RESNET50"
LEARNING_RATE = 0.001

if not os.path.exists("Model/{MODEL_NAME}"):
    os.system(f"mkdir Model/{MODEL_NAME}")

print("[INFO] Hyperparameter:")
print("Epoch: " + str(EPOCHS))
print("Learning rate: " + str(LEARNING_RATE))
print("Batch Size: " + str(BS))

# directory of train and test dataset
train_data_path = f"augm/dataset/train"
list_training = list(os.listdir(train_data_path))
val_data_path = f"augm/dataset/val"
list_validation = list(os.listdir(val_data_path))

label_size = len(list_training)
print("Label Size:", label_size)
image_per_label = len(os.listdir(f"{train_data_path}/{list_training[0]}"))
print("Image Per Label:", image_per_label)

# READ TRAIN AND TEST DATA
train_dataset, train_class_names = optimized_data(train_data_path, BS, train=True)
val_dataset, val_class_names = optimized_data(val_data_path, BS, train=False)

# DEFINE MODEL
model = resnet50(label_size)

# SUMMARY MODEL
model.summary()

# COMPILE MODEL
adam = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("-- MODEL COMPILED --\n")

# Initiate Callbacks
my_callbacks = my_callbacks(MODEL_NAME, label_size, image_per_label, EPOCHS, BS)

# Initiate start time
start_time = time.time()

# TRAIN MODEL
Historia = model.fit(
    train_dataset,
    validation_data=val_dataset,
    callbacks=my_callbacks,
    epochs=EPOCHS,
    batch_size=BS)

# end time
print("--- %s seconds ---" % (time.time() - start_time))

# EVALUATE MODEL
print("\n-- EVALUATE MODEL --")
model = load_my_model(MODEL_NAME, label_size, image_per_label, EPOCHS, BS)

# predict generator
predictions = np.argmax(model.predict(val_dataset), axis=1)
tes = val_dataset.classes

# Print classification report
print_class_reports(tes, predictions, val_class_names, \
    MODEL_NAME, label_size, image_per_label, EPOCHS, BS)

# print confusion matrix
print_conf_matrix(tes, predictions, val_class_names, \
    MODEL_NAME, label_size, image_per_label, EPOCHS, BS)

# OBSERVE MODEL
print("\n-- OBSERVE MODEL --")

# accuracy
visualize_accuracy(Historia, MODEL_NAME, label_size, image_per_label, EPOCHS, BS)

# loss
visualize_loss(Historia, MODEL_NAME, label_size, image_per_label, EPOCHS, BS)
