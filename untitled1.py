
from tkinter import *
from tkinter import filedialog
import cv2
import pickle
import os
import numpy as np
model = pickle.load(open('class.pkl', 'rb'))

def model_predict(img_path):
        print ("Image : ",img_path)
        data1=[]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
        data1.append(img)
        data1 = np.array(data1)
        dataset_size1 = data1.shape[0]
        data1 = data1.reshape(dataset_size1,-1)
        preds = model.predict(data1)
        print("PREDICTION OUTPUT= ",preds[0])
        disease_class = ['NON DEGRADABLE', 'DEGRADABLE']
        remedy="this is it"
        a = preds[0]
        print('Prediction:', disease_class[a])
        print(' ')
        result=disease_class[a]
       
     

model_predict("./n1.jpg")



model_predict("./n2.png")


model_predict("./d1.jpg")


model_predict("./d2.jpg")