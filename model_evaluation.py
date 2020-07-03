# keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.utils import np_utils

import itertools
# timestamp
from time import time

# numpy and sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import argparse

# plots
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

img_width, img_height = 205, 246
evaluation_data_dir = "/content/drive/My Drive/Hackaton/DataFinal/combined"
batch_size = 4
model = load_model('/content/drive/My Drive/Hackaton/nets/final_0922_17fp.h5')
train_datagen = ImageDataGenerator(
        rescale=1. / 255)
validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=False)


y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(y_pred, axis=-1)

classes = validation_generator.classes

cm = confusion_matrix(classes, y_pred)
cr = classification_report(classes, y_pred)
print(cr)
plot_confusion_matrix(cm, ["bad_piece", "good_piece"])
print(validation_generator.class_indices)