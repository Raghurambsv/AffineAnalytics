#importing necessary packages
import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time
from keras.models import load_model
from keras.applications.resnet import ResNet50
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import NASNetLarge




#setting train, validation and test dataset path
train_data_path = 'sanket/train'
validation_data_path = 'sanket/valid'
test_data_path = 'sanket_test/test'


#Setting some parameters for training the model 
batch_size = 8
NUM_CLASSES = 1
img_rows, img_cols = 512, 512
img_channels = 3



#loading Resnet50 from keras application with imagenet weights
base_model = ResNet50(include_top=False, weights="imagenet")


#adding few more layers to make the model work according to current problem statement
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = (Dropout(0.5))(x)

# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)




#join all the layers and create a fully connected layer
model = Model(inputs=base_model.input, outputs=predictions)




#Setting train, validation and test Image generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary')


validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_path,
    shuffle=False,
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    shuffle=False,
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary')



#Initializing own performance metric

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



#Setting callback parameters for model saving, early stopping and reduce_lr

#Saving the best model
model_checkpoint = ModelCheckpoint('resnet50_Png_sanketaddedata.model',monitor='f1', 
                                   mode = 'max', save_best_only=True, verbose=2)

#creating a tensorboard directory
log_dir = './tf-log/newdata_withlr_nodrop'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
early_stopping = EarlyStopping(monitor='f1', mode = 'max',patience=15, verbose=2)
reduce_lr = ReduceLROnPlateau(monitor='f1', mode = 'max',factor=0.5, patience=5, min_lr=0.000001, verbose=2)
cbks = [tb_cb,model_checkpoint,early_stopping,reduce_lr]
#setting optimizer
opt = Adam(lr = 1e-3)


#compiling the model
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=["accuracy",f1])




#started training of the model
model.fit_generator(train_generator,epochs=100,validation_data=validation_generator,callbacks=cbks)


#loading best saved model
model = load_model('./resnet50_Png_sanketaddedata.model',custom_objects={'f1':f1})



#evaluating the model on validation dataset
print(model.evaluate_generator(validation_generator))

#predicting on validation dataset
predict=model.predict_generator(validation_generator)




#classifying images on the basis of probability
prediction = []

for predi in predict:
    if predi>.5:
        prediction.append(1)
    else:
        prediction.append(0)



true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())   

#printing the classification report for all classes 
import sklearn.metrics as metrics
report = metrics.classification_report(true_classes, prediction, target_names=class_labels)
print(report)  


#printing confusion matrix
report = metrics.confusion_matrix(true_classes, prediction)
print(report)


print("test now")

#evaluating the model on test dataset
print(model.evaluate_generator(test_generator))

#predicting on test dataset
predict=model.predict_generator(test_generator)

#classifying images on the basis of probability
prediction = []

for predi in predict:
    if predi>.5:
        prediction.append(1)
    else:
        prediction.append(0)


true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())   

#printing the classification report for all classes 
import sklearn.metrics as metrics
report = metrics.classification_report(true_classes, prediction, target_names=class_labels)
print(report)  


#printing confusion matrix
report = metrics.confusion_matrix(true_classes, prediction)
print(report)

