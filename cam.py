from flask import Blueprint, render_template, redirect, url_for, request
import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os,sys
from gtts import gTTS

def show():
    global word_dict
    model = keras.models.load_model(r"PREDICTION_MODEL.h5")
    word_dict ={0:'zero',1:'one',2:'Two',3:'Three',4:'Four',5:'five',6:'Born',7:'Good',8:'Month',9:'Morning',10:'A',11:'B',12:'C',13:'D',14:'E'}

    global background 
    accumulated_weight = 0.5

    ROI_top = 100
    ROI_bottom = 300
    ROI_right = 150
    ROI_left = 350

    def cal_accum_avg(frame, accumulated_weight):

        global background
        background=None
        
        if background is None:
            background = frame.copy().astype("float")
            return None

        cv2.accumulateWeighted(frame, background, accumulated_weight)

    def segment_hand(frame, threshold=25):
        global background
        
        diff = cv2.absdiff(background.astype("uint8"), frame)

        _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        #Fetching contours in the frame (These contours can be of hand or any other object in foreground) …

        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # If length of contours list = 0, means we didn't get any contours ...
        if len(contours) == 0:
            return None
        else:
            # The largest external contour should be the hand 
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            
            # Returning the hand segment(max contour) and the thresholded image of hand...
            return (thresholded, hand_segment_max_cont)
        

    cam = cv2.VideoCapture(0)
    num_frames =0
    count=0
    output=[0]*20
    while True:
        ret, frame = cam.read()

        # flipping the frame to prevent inverted image of captured frame...
    
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        # ROI from the frame
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


        if num_frames < 70:
        
            cal_accum_avg(gray_frame, accumulated_weight)
        
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",(80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
        else: 
            # segmenting the hand region
            hand = segment_hand(gray_frame)
        
            # Checking if we are able to detect the hand...
            if hand is not None:
                global pred
                global text1
                thresholded, hand_segment = hand

            # Drawing contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right,ROI_top)], -1, (255, 0, 0),1)
            
                cv2.imshow("Thesholded Hand Image", thresholded)
            
                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))
                pred = model.predict(thresholded)
                text1=word_dict[np.argmax(pred)]
                cv2.putText(frame_copy, word_dict[np.argmax(pred)],(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            
            
    # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,ROI_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
        num_frames += 1

    # Display the frame with segmented hand
        cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _",(10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        cv2.imshow("Sign Detection", frame_copy)


    # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

 # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()


def speak():
    fh= open("STORAGE.txt","r")
    myText=fh.read().replace("\n"," ")

    language = 'en'

    output =gTTS(text=myText, lang=language, slow=False)

    output.save("output.mp3")

    fh.close()
    os.system("output.mp3")    

def saveFile():
    if(text1!=None):
        print("NEW RESULT")
        print()
        print(text1)
        if(len(text1)==1):
            with open('STORAGE.txt','a') as f:
                f.write(text1)
                f.close()
        else:
            with open('STORAGE.txt','a') as f:
                f.write("  ")
                f.write(text1)
                f.write("  ")
                f.close()


def clear():
    open('STORAGE.txt','w').close()




  


