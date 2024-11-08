import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
import pickle

#from imutils import paths
import os
import glob
import cv2
import numpy as np

def getListOfFiles(dirName):

    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

imagePaths = getListOfFiles("./datasets/") 

data = []
lables = []
c = 0 ## to see the progress
print("Preprocessing Images")
for image in imagePaths:

    lable = os.path.split(os.path.split(image)[0])[1]
    lables.append(lable)

    img = cv2.imread(image)
    img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
    data.append(img)
    c=c+1
    #print(c)

print("Preprocessing Done.....")

data = np.array(data)
lables = np.array(lables)

le = LabelEncoder()
lables = le.fit_transform(lables)

myset = set(lables)
print(myset)

dataset_size = data.shape[0]
data = data.reshape(dataset_size,-1)

print(data.shape)
print(lables.shape)
print(dataset_size)

(trainX, testX, trainY, testY ) = train_test_split(data, lables, test_size= 0.25, random_state=42)

model = RandomForestClassifier()
print("TRAINING .....")
model.fit(trainX, trainY)
print("TRAINING Done.....SAVING MODEL")
knnpickle=open('classrandom.pkl','wb')
pickle.dump(model,knnpickle)

print("GENERATING REPORT.....")
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

