# -*- coding: utf-8 -*-

# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import pickle
import os
import numpy as np
from skimage import io
model = pickle.load(open('classrandom.pkl', 'rb'))



def model_predict(img_path):
        print ("Image : ",img_path)
        data1=[]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
        data1.append(img)
        data1 = np.array(data1)
        dataset_size1 = data1.shape[0]
        data1 = data1.reshape(dataset_size1,-1)
        preds = model.predict(data1)
        print("PREDICTION OUTPUT= ",preds[0])
        clss = ['NON DEGRADABLE', 'DEGRADABLE']
        a = preds[0]
        print('Prediction:', clss[a])
        print(' ')
        return clss[a]
        
      

def select_image():
	global panelA, panelB

	path = filedialog.askopenfilename()
    	# ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path)
		image=cv2.resize(image,(600,500)) 
		c=model_predict(path)
		print(c)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
        
        		# if the panels are None, initialize them
		if panelA is None :
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=400, pady=10)

		else:
			# update the pannels
			panelA.configure(image=image)
			
			panelA.image = image
			
            
		Btn1.configure(text="DETECTED: "+c)


            
root = Tk()
root.state('zoomed')
root['bg']='blue'

lbl2 = Label(root, text = "WASTE TYPE DETECTION",font=("Arial", 15))
lbl2.pack()

panelA = None
panelB = None
outlabel=None


btn = Button(root, text="Select an image",font=("Arial", 15), command=select_image)
btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")
Btn1 = Button(root, text="",font=("Arial", 25), bg='blue', fg='yellow')
Btn1.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")


root.mainloop()