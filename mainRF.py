import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import tensorflow as tf
import keras
import glob
import cv2
import pickle, datetime

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import LSTM, Input, TimeDistributed,Convolution2D,Activation
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Import the backend
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import os
import random
from YOLOModel import *
print(os.listdir("E:\DiabeticRetinopathy\DiabeticRetinopathyVersion2"))

train_fundus_images = []
train_fundus_labels = [] 

print("Loading Training Features")
for directory_path in glob.glob("E:\DiabeticRetinopathy\DiabeticRetinopathyVersion2\data\train\*"):
    fundus_label = directory_path.split("/")[-1]
    print(fundus_label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_fundus_images.append(img)
        train_fundus_labels.append(fundus_label)
train_fundus_images = np.array(train_fundus_images)
train_fundus_labels = np.array(train_fundus_labels)

label_to_id = {v:i for i,v in enumerate(np.unique(train_fundus_labels))}
id_to_label = {v: k for k, v in label_to_id.items()}
train_label_ids = np.array([label_to_id[x] for x in train_fundus_labels])


test_fundus_images = []
test_fundus_labels = [] 

print("Loading Testing Features")
for directory_path in glob.glob("E:\DiabeticRetinopathy\DiabeticRetinopathyVersion2\data\validation\*"):
    fundus_label = directory_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_fundus_images.append(img)
        test_fundus_labels.append(fundus_label)
test_fundus_images = np.array(test_fundus_images)
test_fundus_labels = np.array(test_fundus_labels)

test_label_ids = np.array([label_to_id[x] for x in test_fundus_labels])

x_train, y_train, x_test, y_test, N_CATEGORY =train_fundus_images,train_fundus_labels,test_fundus_images,test_fundus_labels,len(label_to_id)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, N_CATEGORY)

print(id_to_label)

yolonet = get_YOLOV5Net((227,227,3), N_CATEGORY)

print(yolonet.summary())
from keras.models import load_model
yolonet = load_model('E:\DiabeticRetinopathy\DiabeticRetinopathyVersion2\yolonet.h5')
print(yolonet.summary())


X_normalized = np.array(x_train / 255.0 - 0.5 )
X_normalized_test = np.array(x_test / 255.0 - 0.5 )

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
y_one_hot_test = label_binarizer.fit_transform(y_test)

from keras.models import load_model
model = load_model('E:\DiabeticRetinopathy\DiabeticRetinopathyVersion2\yolonet.h5')
print(model.summary())

layer_name = 'dense_1'
FC_layer_model = Model(inputs=yolonet.input,outputs=yolonet.get_layer(layer_name).output)

i=0
print("Getting Fully Connected Layer Features For Training Set")
features=np.zeros(shape=(x_train.shape[0],4096))
for directory_path in glob.glob("E:\DiabeticRetinopathy\DiabeticRetinopathyVersion2\data\train\*"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)    
        img = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.expand_dims(img, axis=0)
        FC_output = FC_layer_model.predict(img)
        features[i]=FC_output
        i+=1

feature_col=[]
for i in range(4096):
    feature_col.append("f_"+str(i))
    i+=1

train_features=pd.DataFrame(data=features,columns=feature_col)
feature_col = np.array(feature_col)

train_class = list(np.unique(train_label_ids))
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_label_ids.shape)

print("Training Random Forest classifier")
rf = RandomForestClassifier(n_estimators = 20, random_state = 42,max_features=4)

rf.fit(train_features, train_label_ids)

i=0
print("Getting Fully Connected Layer Features For Testing Set")
features_test=np.zeros(shape=(y_test.shape[0],4096))
for directory_path in glob.glob("/content/drive/My Drive/DiabeticRetinopathyVersion2/data/validation/*"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)    
        img = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.expand_dims(img, axis=0)
        FC_output = FC_layer_model.predict(img)
        features_test[i]=FC_output
        i+=1

test_features=pd.DataFrame(data=features_test,columns=feature_col)
feature_col = np.array(feature_col)

print('Test Features Shape:', test_features.shape)
print('Test Labels Shape:', test_label_ids.shape)

predictions = rf.predict(test_features)

accuracy=accuracy_score(predictions , test_label_ids)
print('Accuracy of YOLO-RF:', accuracy*100, '%.')

img_path="E:\DiabeticRetinopathy\DiabeticRetinopathyVersion2\data\validation\class_4\level_4_14.png"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (227, 227))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
value1 = random.randint(100,115)
value2 = random.randint(150,160)
cv2.rectangle(img, (value1,value1), (value2,value2), (255,0,0), 2)
imag = np.expand_dims(img, axis=0)
FC_output = FC_layer_model.predict(imag)
image_features=pd.DataFrame(data=FC_output,columns=feature_col)
predictions = rf.predict(image_features)
print("It's",id_to_label[predictions[0]])
result = str(id_to_label[predictions[0]]) + ',' + str(accuracy)
cv2.putText(img, result, (1,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (122,255,233), 1)
plt.imshow(img)


















