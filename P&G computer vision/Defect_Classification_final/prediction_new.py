#importing neccesary packages
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time
from keras import backend as K

#defining custom funtions
start = time.time()
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#loading model
model = load_model('./resnet50_1_var_lr.model',custom_objects={'f1':f1})



from keras.preprocessing.image import ImageDataGenerator

#setting up test folder path
test_data_path = './test'

#setting test data generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    shuffle=False,
    target_size=(512, 512),
    class_mode=None,)


test_generator.reset()


#predicting 
pred=model.predict_generator(test_generator,verbose=1,steps=1)


#classifying images on the basis of probability
prediction = []

for predi in predict:
    if predi>.5:
        prediction.append(1)
    else:
        prediction.append(0)