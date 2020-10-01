# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:51:16 2020

@author: ACER
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten, BatchNormalization, Dropout
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

model = Sequential()
model.add(Conv2D(128,(3,3),input_shape = (64,64,1),activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units = 256, activation = 'sigmoid'))
model.add(BatchNormalization())
model.add(Dense(units = 7, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

train_gen = ImageDataGenerator(rescale = 1./255)
test_gen = ImageDataGenerator(rescale = 1./255)

training_set = train_gen.flow_from_directory(r"C:\Users\ACER\Desktop\AI\Facial Emotion Rec\images\train",target_size = (64,64),color_mode="grayscale",batch_size = 128,class_mode = 'categorical')
validation_set = test_gen.flow_from_directory(r"C:\Users\ACER\Desktop\AI\Facial Emotion Rec\images\validation",target_size = (64,64),color_mode="grayscale",batch_size = 128,class_mode = 'categorical')

model.fit_generator(training_set,steps_per_epoch = 706,epochs = 5,validation_data = validation_set,validation_steps = 79)

model.save("emotion_model.h5")

test_image = image.load_img(r"C:\Users\ACER\Desktop\AI\Facial Emotion Rec\images\validation\sad\1173.jpg", color_mode = "grayscale",target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = model.predict(test_image)
print(result)
print(training_set.class_indices)