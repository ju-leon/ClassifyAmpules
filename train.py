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

# timestamp
from time import time

# numpy and sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import argparse

# plots
import itertools
import matplotlib.pyplot as plt


# method for building the neural network architecture
def get_classifier(input_shape):
    model = Sequential()

    model.add(Conv2D(128, (5, 5), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(265, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1048))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(2))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, clipvalue=0.1),
                  metrics=['accuracy'])
    return model


# method for defining the training callbacks
def get_callbacks():
    return [TensorBoard(log_dir='logs/{}'.format(time())),
            ModelCheckpoint(filepath="/content/drive/My Drive/Hackaton/nets/leon.h5", monitor='val_accuracy',
                            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]


# method for plotting the confusion matrix
def plot_confusion_matrix(validation_generator):
    model = load_model('/content/drive/My Drive/Hackaton/nets/final.h5')

    y_pred = model.predict_generator(validation_generator)
    y_pred = np.argmax(y_pred, axis=-1)

    classes = validation_generator.classes

    cm = confusion_matrix(classes, y_pred)
    cr = classification_report(classes, y_pred)
    print(cr)

    plot_confusion_matrix_pretty(cm, classes=["bad_piece", "good_piece"])

    print(validation_generator.class_indices)

def plot_confusion_matrix_pretty(cm, classes,
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


def main():
    parser = argparse.ArgumentParser(description='Preprocess raw images by intelligent downscaling')

    parser.add_argument('data_dir')

    args = parser.parse_args()

    # constants for training
    img_width, img_height = 205, 246
    train_data_dir = args.data_dir
    nb_train_samples = 892
    nb_validation_samples = 382
    epochs = 100
    batch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # augmentation used for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3)

    # train and validation generator
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False)

    classifier = get_classifier(input_shape)

    classifier.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=get_callbacks())

    plot_confusion_matrix(validation_generator)



if __name__ == "__main__":
    main()
