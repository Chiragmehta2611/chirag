from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
IMAGE_SIZE = [224, 224]
from google.colab import drive
drive.mount('/content/drive')
!ls '/content/drive'
train_path = '/content/drive/MyDrive/AI_MAJOR/dataset/training'
test_path = '/content/drive/MyDrive/AI_MAJOR/dataset/testing'

from PIL import Image 
import os 
from IPython.display import display
from IPython.display import Image as _Imgdis 

  
folder = train_path+'/flowers'


onlyflowersfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print("Working with {0} images".format(len(onlyflowersfiles)))
print("Image examples: ")


for i in range(10):
    print(onlyflowersfiles[i])
     display(_Imgdis(filename=folder + "/" + onlyflowersfiles[i], width=240, height=240))

Xception = Xception(input_shape = IMAGE_SIZE + [3] , weights = 'imagenet' ,include_top = False)
Xception.output
Xception.input
for layer in Xception.layers:
  layer.trainable = False
  folders = glob('/content/drive/MyDrive/AI_MAJOR/dataset/training/*')
print(len(folders))
x = Flatten()(Xception.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=Xception.input, outputs=prediction)
model.summary()

from keras import optimizers

adam = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

from datetime import datetime
from keras.callbacks import ModelCheckpoint


checkpoint = ModelCheckpoint(filepath='xception_model.hdf5', 
                               verbose=2, save_best_only=True)

callbacks = [checkpoint]

start = datetime.now()
batch_size = 32
model_history=model.fit(train_set, validation_data=test_set, epochs=25, 
                        steps_per_epoch=int(550/batch_size), validation_steps=int(288/batch_size),
                        callbacks=callbacks ,verbose=2)

duration = datetime.now() - start
print("Training completed in time: ", duration)

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('CNN Model accuracy values')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()