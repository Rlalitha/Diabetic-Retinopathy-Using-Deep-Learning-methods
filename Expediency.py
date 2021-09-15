
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os 
import warnings
import csv
import time
from skimage.transform import resize
import cv2
import shutil
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import json
d = 0.5
e = 0.95
from tqdm import tqdm

class Configuration(object):
    IMAGES_PATH = r'E:\Version_2\train_data\*'
    
class Data_Loader(Configuration):

    def __init__(self,data_Path = None):

        self.data_Path = self.IMAGES_PATH
        self.images = []
        self.labels = []

    def resize_Image(self,image):
        return cv2.resize(image,(1050,1050))

    def transform_Image(self,image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    def load_Images(self,debug=True,load=True):
        if load == True:
            self.image_Array = np.load('image_Array.npy')
            self.label_Array = np.array('label_Array.npy')

        else:
            data_Set = glob(self.data_Path)
            
            for data in data_Set:
                image_Path = glob(data+'\*.png')
                if str(data[-7:]) == "class_0":
                    label = 0
                if str(data[-7:]) == "class_1":
                    label = 1
                if str(data[-7:]) == "class_2":
                    label = 2
                if str(data[-7:]) == "class_3":
                    label = 3
                if str(data[-7:]) == "class_4":
                    label = 4
                count = 0
                for image in image_Path:
                    print(image)
                    image = cv2.imread(image)
                    image = self.resize_Image(image=image)
                    image = self.transform_Image(image=image)
                    self.images.append(image)
                    self.labels.append(label)
                    count = count+1
                    if count >= 10:
                        break

            self.image_Array = np.array(self.images)
            self.label_Array = np.array(self.labels)

        if (debug):

            print("The shape of the default image is ",self.image_Array[0].shape)
            plt.imshow(self.image_Array[0])
            plt.show()

        return self.image_Array, self.label_Array

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM_Classifier(object):
    def __init__(self,X,y):
      self.X = X
      nsamples, nx, ny,nz = self.X.shape
      self.X = self.X.reshape((nsamples,nx*ny*nz))
      self.y = y

    def create_Classifier(self):
      parameters = {'kernel':('linear', 'rbf'), 'C':[1, 20]}
      self.classifier = SVC()
      self.classifier = GridSearchCV(self.classifier, parameters)
    def data_Splitting(self):
      self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y)

    def train(self):
      print(self.classifier)
      self.classifier.fit(self.X_train,self.y_train)

    def evaluate(self):
      self.y_pred = self.classifier.predict(self.X_test)
      plot_confusion_matrix(self.classifier, self.X_test, self.y_test)
      plt.show()
      return accuracy_score(self.y_pred,self.y_test)


from sklearn.ensemble import RandomForestClassifier

class RF_Classifier(object):
    def __init__(self,X,y):
      self.X = X
      nsamples, nx, ny,nz = self.X.shape
      self.X = self.X.reshape((nsamples,nx*ny*nz))
      self.y = y

    def create_Classifier(self):
      self.classifier = RandomForestClassifier()
      param_grid = {
                'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': [2, 3],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 200, 300, 1000]
                }
      self.classifier = GridSearchCV(estimator = self.classifier, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    def data_Splitting(self):
      self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y)

    def train(self):
      print(self.classifier)
      self.classifier.fit(self.X_train,self.y_train)

    def evaluate(self):
      self.y_pred = self.classifier.predict(self.X_test)
      plot_confusion_matrix(self.classifier, self.X_test, self.y_test)
      plt.show()
      return accuracy_score(self.y_pred,self.y_test)


from sklearn.tree import DecisionTreeClassifier

class DT_Classifier(object):
    def __init__(self,X,y):
      self.X = X
      nsamples, nx, ny,nz = self.X.shape
      self.X = self.X.reshape((nsamples,nx*ny*nz))
      self.y = y

    def create_Classifier(self):
      sample_split_range = list(range(2, 10)) 
      parameters = dict(min_samples_split=sample_split_range)
      self.classifier = DecisionTreeClassifier()
      self.model = GridSearchCV(self.classifier, param_grid=parameters, cv=10,verbose=1,n_jobs=-1)
      

    def data_Splitting(self):
      self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y)

    def train(self):
      print(self.classifier)
      self.classifier.fit(self.X_train,self.y_train)

    def evaluate(self):
      self.y_pred = self.classifier.predict(self.X_test)
      plot_confusion_matrix(self.classifier, self.X_test, self.y_test)
      plt.show()
      return accuracy_score(self.y_pred,self.y_test)

class DensNet(object):
    @staticmethod
    def DensNet(X,y):
        X = X
        y = to_categorical(y)
        trainx,testx,trainy,testy=train_test_split(X,y,test_size=0.2,random_state=44)
        datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                        width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")
        densenet_weights_path = 'Saved_Models/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(1050,1050,3),include_top=False,weights=densenet_weights_path,pooling='avg')
        pretrained_model3.trainable = False
        inputs3 = pretrained_model3.input
        x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)
        outputs3 = tf.keras.layers.Dense(5, activation='softmax')(x3)
        model = tf.keras.Model(inputs=inputs3, outputs=outputs3)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        his=model.fit(datagen.flow(trainx,trainy,batch_size=16),validation_data=(testx,testy),epochs=2)
        get_acc = his.history['accuracy']
        value_acc = his.history['val_accuracy']
        get_loss = his.history['loss']
        validation_loss = his.history['val_loss']

        epochs = range(len(get_acc))
        plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
        plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
        plt.title('Training vs validation accuracy')
        plt.legend(loc=0)
        plt.figure()
        plt.show()


def adjust_gamma(image, gamma=1.0):
   table = np.array([((i / 255.0) ** gamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

def extract_ma(image):
    r,g,b=cv2.split(image)
    comp=255-g
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    histe=clahe.apply(comp)
    adjustImage = adjust_gamma(histe,gamma=3)
    comp = 255-adjustImage
    J =  adjust_gamma(comp,gamma=4)
    J = 255-J
    J = adjust_gamma(J,gamma=4)
    K=np.ones((11,11),np.float32)
    L = cv2.filter2D(J,-1,K)
    
    ret3,thresh2 = cv2.threshold(L,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel2=np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3=np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    return opening

def extract_bv(image):		
	b,green_fundus,r = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(green_fundus)
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
	f5 = clahe.apply(f4)		
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
	contours,_ = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 200:
			cv2.drawContours(mask, [cnt], -1, 0, -1)			
	im = cv2.bitwise_and(f5, f5, mask=mask)
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	
	fundus_eroded = cv2.bitwise_not(newfin)	
	xmask = np.ones(green_fundus.shape[:2], dtype="uint8") * 255
	xcontours,_  = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
	for cnt in xcontours:
		shape = "unidentified"
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
		if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
			shape = "circle"	
		else:
			shape = "veins"
		if(shape=="circle"):
			cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
	finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
	blood_vessels = cv2.bitwise_not(finimage)
	return blood_vessels

def plot_Metrics(jsonFile):
    with open(jsonFile, 'r') as openfile:
       json_object = json.load(openfile)
    
    classifiers = list(json_object.keys())
    values = list(json_object.values())
      
    fig = plt.figure(figsize = (10, 5))
    plt.bar(classifiers, values, color ='maroon',
            width = 0.4)
     
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Chart")
    plt.show()



