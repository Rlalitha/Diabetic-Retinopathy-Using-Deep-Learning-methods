import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import PySimpleGUI as sg
from PIL import Image
import argparse
import time
from pathlib import Path
from Expediency import *
net = cv2.dnn.readNetFromDarknet("yolov5_custom.cfg",r"yolov5_custom_last(1).weights")

classes = ['NO DR','Mild DR','Moderate DR','Severe DR','Proliferative DR']


def detect(image):
    img = cv2.imread(image)
    img = cv2.resize(img,(1280,720))
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []
    label = None
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)

                x = int(center_x - w/2)
                y = int(center_y - h/2)



                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label + " " + confidence, (x,y-10),font,2,color,2)


    cv2.imshow('img',img)
    cv2.imwrite("sample.jpg",img)
    cv2.waitKey(0)
    if label is None:
        result = 'No DR'
    else:
        result = label
    return result,img

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*"),
              ("PNG (*.png)","*.png")]

def open_window(image):
    layout = [
        [sg.Text("Diagnosis Result")],
        [sg.Image(key="-IMAGE-")]]
    
    window = sg.Window("DR Diagnosis", layout, modal=True,finalize=True)
    image = Image.fromarray(image)
    image.thumbnail((400, 400))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    window["-IMAGE-"].update(data=bio.getvalue())
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()

def second_window(image):
    layout = [
        [sg.Text("Microneurysm Detection")],
        [sg.Image(key="-IMAGE-")]]
    
    window = sg.Window("DR Diagnosis", layout, modal=True,finalize=True)
    image = Image.fromarray(image)
    image.thumbnail((400, 400))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    window["-IMAGE-"].update(data=bio.getvalue())
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()

def main_window():
    layout = [[sg.Text("User Name  "),sg.Input("")],
              [sg.Text("Password    "),sg.Input("", password_char='*')],
              [sg.Button("Login",key="open")]]
    window = sg.Window("Login Window", layout,size=(400,200), element_justification='c')
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "open":
            if values[0] == 'Lalitha':
                if values[1] == 'diabeticretinopathy':
                    window.close()
                    person_window()
                else:
                    sg.Popup("Invalid Password")
            else:
                sg.Popup("Invalid User Name")
                    
def person_window():
    layout = [[sg.Text("Patient Details Window")],
              [sg.Text("Patient Name  "),sg.Input("",size=(30,1))],
              [sg.Text("Age               "),sg.Input("", size=(30,1))],
              [sg.Text("Description     "),sg.Input("", size=(30,1))],
              [sg.Button("Enter",key="open")]]
    window = sg.Window("Details Window", layout,size=(400,200), element_justification='c')
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "open":
            if values[0] == '' or values[1] == '' or values[2] == '':
                sg.Popup("Please Provide the details")
            else:
                window.close()
                main()
def main():
    layout = [
        [sg.Text("Load Input Image for DR Diagnosis")],
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
        [sg.Text("",size=(25, 1), key="-F-")],
        [sg.Button("Predict")],
        [sg.Button("Back")],
    ]
    window = sg.Window("Diabetic Retinopathy Detection", layout,size=(800,600), element_justification='c')
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())

        if event == "Back":
            window.close()
            person_window()  
                

        if event == "Predict":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                result,image = detect(image=filename)
                if result == "level_0":
                    result = "The Diagnosis Result is No DR"
                    
                if result == "level_1":
                    result = "The Diagnosis Result is Non Proliferative DR"
                    image = cv2.imread(filename)
                    image2 = extract_ma(image)
                    image = extract_bv(image)
                    window["-F-"].update(result)
                    second_window(image2)
                if result == "level_2":
                    result = "The Diagnosis Result is Moderate" 
                if result == "level_3":
                    result = "The Diagnosis Result is Severe"
                if result == "level_4":
                    result = "The Diagnosis Result is Proliferative DR"
                window["-F-"].update(result)
                open_window(image)
                
    window.close()
    
if __name__ == "__main__":
    main_window()
