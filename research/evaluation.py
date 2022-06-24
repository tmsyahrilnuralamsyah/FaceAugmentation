import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Font option
title_font = {'weight': 'medium', 'size': 'medium'}
axis_font = {'size': 'small'}

# Visualize accuracy from training model
def visualize_accuracy(Historia, MODEL_NAME, label_size, image_per_label, EPOCHS, BS):
    # xlimit
    Numero = np.arange(1, len(Historia.history["accuracy"])+1, 1)

    # plot accuracy
    plt.figure()
    plt.plot(Numero, Historia.history["accuracy"], label="train_acc")
    plt.plot(Numero, Historia.history["val_accuracy"], label="val_acc")
    plt.title("Training and Validation Accuracy on Dataset", **title_font)
    plt.xlabel(
        f"Epoch {EPOCHS} Batch Size {BS} Label {label_size} Data {image_per_label}", **axis_font)
    plt.ylabel("Accuracy", **axis_font)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        f"Model/{MODEL_NAME}/figure_accuracy_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}.png", dpi=600)

# Visualize loss from training model
def visualize_loss(Historia, MODEL_NAME, label_size, image_per_label, EPOCHS, BS):
    # xlimit
    Numero = np.arange(1, len(Historia.history["loss"])+1, 1)

    # plot loss
    plt.figure()
    plt.plot(Numero, Historia.history["loss"], label="train_loss")
    plt.plot(Numero, Historia.history["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss on Dataset", **title_font)
    plt.xlabel(
        f"Epoch {EPOCHS} Batch Size {BS} Label {label_size} Data {image_per_label}", **axis_font)
    plt.ylabel("Loss", **axis_font)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid()
    plt.savefig(
        f"Model/{MODEL_NAME}/figure_loss_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}.png", dpi=600)

# Print classification report to csv
def print_class_reports(tes, predictions, target_names, MODEL_NAME, label_size, image_per_label, EPOCHS, BS):
    report = classification_report(
        tes, predictions, target_names=target_names, output_dict=True)
    dataframe = pd.DataFrame(report).transpose()
    dataframe.to_csv(
        f"Model/{MODEL_NAME}/reports_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}.csv")
    print(classification_report(tes, predictions, target_names=target_names))

# print confusion matrix and visualize it
def print_conf_matrix(tes, predictions, target_names, MODEL_NAME, label_size, image_per_label, EPOCHS, BS):
    confusion_mtx = confusion_matrix(tes, predictions)
    plt.figure()
    sns.heatmap(confusion_mtx, xticklabels=target_names, yticklabels=target_names,
                # annot=True,
                fmt='g',
                cbar_kws={'label': 'Individual Image'},
                )
    plt.title("Confusion Matrix on Prediction", **title_font)
    plt.xlabel('Prediction', **axis_font)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.ylabel('Label', **axis_font)
    plt.tight_layout()
    plt.savefig(
        f"Model/{MODEL_NAME}/confusion_matrix_{MODEL_NAME}_label{label_size}_data{image_per_label}_e{EPOCHS}_bs{BS}.png", dpi=600)
    print(confusion_mtx)
