# -*- coding: utf-8 -*-
"""
build AutoEncoder model and cosine similarty to image recogination
"""
import sys
import datetime
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from matplotlib import cm
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
plt.style.use('ggplot')
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#load data set utfaces dataset from kaggle
from google.colab import files
files.upload()
# Let's make sure the kaggle.json file is present.
!ls -lha kaggle.json
# Next, install the Kaggle API client.
!pip install -q kaggle

# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json"

!kaggle datasets download -d jangedoo/utkface-new

#unzip dataset
!unzip utkface-new.zip

from imutils import paths
imagePaths = list(paths.list_images("UTKFace"))
#resize images to 32*32
import cv2
X_data =[]
for file in imagePaths:
    face = cv2.imread(file)
    face = cv2.resize(face, (32, 32) )
    X_data.append(face)
#Normalize data 
X_data=np.array(X_data)/255
#split data to train and test
(trainX, testX) = train_test_split(X_data,
	test_size=0.20, random_state=42)
#identify data shapes 
inputShape=trainX.shape[1:]
inputShape
#using data generatorrator
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
#model archtecture
#--build Encoder
Model_Auto=Sequential()
Model_Auto.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
Model_Auto.add(Activation("relu"))
Model_Auto.add(BatchNormalization())
		#model.add(BatchNormalization(axis=chanDim))
Model_Auto.add(Conv2D(32, (3, 3), padding="same"))
Model_Auto.add(Activation("relu"))
Model_Auto.add(BatchNormalization())
		#model.add(BatchNormalization(axis=chanDim))
Model_Auto.add(MaxPooling2D(pool_size=(2, 2)))
Model_Auto.add(Dropout(0.3))
		# second CONV => RELU => CONV => RELU => POOL layer set
Model_Auto.add(Conv2D(64, (3, 3), padding="same"))
Model_Auto.add(Activation("relu"))
Model_Auto.add(BatchNormalization())
Model_Auto.add(Conv2D(64, (3, 3), padding="same"))
Model_Auto.add(Activation("relu"))
Model_Auto.add(BatchNormalization())
Model_Auto.add(MaxPooling2D(pool_size=(2, 2)))
#build decoder
#at this point use decode
Model_Auto.add(Conv2D(64, (3, 3), padding="same"))
Model_Auto.add(Activation("relu"))
Model_Auto.add(BatchNormalization())
Model_Auto.add(UpSampling2D((2, 2)))
Model_Auto.add(Conv2D(64, (3, 3), padding="same"))
Model_Auto.add(Activation("relu"))
Model_Auto.add(BatchNormalization())
Model_Auto.add(Conv2D(32, (3, 3), padding="same"))
Model_Auto.add(BatchNormalization())
Model_Auto.add(Activation("relu"))
Model_Auto.add(UpSampling2D((2, 2)))
Model_Auto.add(BatchNormalization())
Model_Auto.add(Conv2D(3, (3, 3), padding="same"))
Model_Auto.add(Activation("sigmoid"))

Model_Auto.compile(loss="binary_crossentropy", optimizer='sgd')

len(trainX),len(testX)
#fit without image generator
hist_Fac=Model_Auto.fit(trainX,trainX,
                        batch_size=42,
    epochs=100,
    validation_data=(testX, testX))
#show loss and val_loss
loss = hist_Fac.history['loss']
val_loss = hist_Fac.history['val_loss']
epochs = range(100)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#predict test data
decoded_Facs = Model_Auto.predict(testX)
#figure some images  from test to comapre with predict
plt.figure(figsize=(10, 2))
# for i in range(n):
    # display original
plt.imshow(testX[1].reshape(32, 32,3))
plt.gray()
#figure some images  from  predict
plt.figure(figsize=(10, 2))
# for i in range(n):
    # display original
plt.imshow(decoded_Facs[1].reshape(32, 32,3))
plt.gray()

# from here build cosine similarty
List_Mg_Pred=[]
#function to predict images and store predict in list of class by name and binary images
def Pred_imge(imge):
  newdata_ashraf=[]
  ashraf_imge = cv2.imread(imge)
  nameimg=imge.split('.')[0]
  ashraf_imge = cv2.cvtColor(ashraf_imge, cv2.COLOR_BGR2RGB)
  ashraf_imge = cv2.resize(ashraf_imge, (32, 32))
  newdata_ashraf.append(ashraf_imge)
  newdata_ashraf = np.array(newdata_ashraf) / 255.0  
  ashraf_Facs = Model_Auto.predict(newdata_ashraf)
  List_Mg_Pred.append(Img_class(nameimg,ashraf_Facs))#name and image predicted
  #function to compare between real image and list of predicted images using cosine similarty function
def Consin_Sim(OrginlImg):
  newdata_Mariam=[]
  Reslt=""
  Mariam_imge = cv2.imread(OrginlImg)
  Mariam_imge = cv2.cvtColor(Mariam_imge, cv2.COLOR_BGR2RGB)
  Mariam_imge = cv2.resize(Mariam_imge, (32, 32))
  newdata_Mariam.append(Mariam_imge)
  newdata_Mariam = np.array(newdata_Mariam) / 255.0
  for img in List_Mg_Pred:
    percnt=Cosine_sm(img.Predicted.reshape((1,3072 )), newdata_Mariam.reshape((1, 3072)))
    print(percnt[0])
    if percnt[0]>0.95:
      Reslt=img.Name
      break;
    else:
      Reslt="UnKnown"

  return Reslt
#loaad images to predict
uploaded = files.upload()
for fn in uploaded.keys():
  Pred_imge(fn)



#class to store name and image binary 
  
class Img_class:  
    def __init__(self, nameimg, Predicted):  
        self.Name = nameimg  
        self.Predicted = Predicted

from sklearn.metrics.pairwise import cosine_similarity as Cosine_sm
 #!rm *.jpeg
#loaad real image to comapre
uploaded = files.upload()
for fn in uploaded.keys():
  print(Consin_Sim(fn))


