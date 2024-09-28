# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:46:14 2021

@author: khosro
"""


#def autoMate(key):
#    pyautogui.keyDown(key)
#    return
#def collision(data):
#    #Check collision for birds
#    for i in range(170, 210):
#        for j in range(209, 374):
#            if data[i, j] < 100:
#                autoMate("down")
#                return
#    #Check collision for cactus
#    for i in range(237, 275):
#        for j in range(376, 440):
#            if data[i, j] < 100:
#                autoMate("up")
#                return
#    return
#if __name__ == "__main__":
#    print("The game is starting in 2 seconds...")
#    time.sleep(2)
#    while True:
#        image = ImageGrab.grab().convert('L')
#        data = image.load()
#        
           
        
        
        
        
        
        
        

import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard
from dinogame import DinoGame
import pygame
from pygame import *
import pyautogui
from PIL import Image, ImageGrab
import time





def autoMate(key):
    pyautogui.press(key)
    return










kernel_size = 5
low_threshold = 50
high_threshold = 150

rho = 3  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 20  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments


def human_skin_detector(img,minr,ming,minb,maxr,maxg,maxb,var ):
    (x,y,z)=np.shape(img)
    red=img[:,:,0]
    green=img[:,:,1]
    blue=img[:,:,2]
    skin_1 = np.bitwise_and(red>minr - var , maxr +var > red)
    skin_2 = np.bitwise_and(green> ming - var , maxg +var > green)
    skin_3 = np.bitwise_and(blue > minb - var,maxb +var > blue)
    a=np.bitwise_and(skin_1,skin_2)
    skin=np.bitwise_and(a,skin_3)
    return skin


cap = cv2.VideoCapture(0)

var=7
s=0
x=100
y=100
w=100
h=100

while 1:
    ret, img = cap.read()
    if s==0:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if keyboard.is_pressed('s'):
            s=1
            sample_skin=img[x:x+w,y:y+h,:]
            minr=np.min(sample_skin[:,:,0])
            ming=np.min(sample_skin[:,:,1])
            minb=np.min(sample_skin[:,:,2])
            maxr=np.max(sample_skin[:,:,0])
            maxg=np.max(sample_skin[:,:,1])
            maxb=np.max(sample_skin[:,:,2])
            print("The game is starting ...")
            autoMate("up")
        closing_w=img
        
    else:
        
        #keyboard.press_and_release('space')
        skin_mask_w=human_skin_detector(img,minr,ming,minb,maxr,maxg,maxb,var)
        skin_mask_w = skin_mask_w.astype(np.uint8)
        skin_mask_w*=255
        kernel = np.ones((11,11),np.uint8)
        mask_w = cv2.morphologyEx(skin_mask_w, cv2.MORPH_OPEN,kernel)
        
        kernel_2 = np.ones((3,3),np.uint8)
        closing_w = cv2.morphologyEx(mask_w , cv2.MORPH_CLOSE, kernel_2)
        
        contours, hierarchy = cv2.findContours(closing_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(hierarchy)
        if hierarchy is not None:
            cnt =  max(contours, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
        #cv2.drawContours(closing, [cnt], 0, (0,255,0), 3)

            cv2.rectangle(closing_w,(x,y),(x+w,y+h),(255,0,0),0)
            crop=closing_w[y+1:y+h,x+1:x+w]
        #gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray=crop
        
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        
        
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        n=0
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)
                    n=n+1
        else:
            n=0
        
        
        #backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        # Draw the lines on the  image
        #lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

        #cv2.imshow('line_image', gray)
        
        
        if n>10 :
            autoMate("up")
#            keyboard.press(Key.up)        
##            delay = random.uniform(0, 2)  
##            time.sleep(delay)
#            keyboard.release(Key.up)
            #keyboard.press_and_release('space')
        
        
        #closing_w=img
        
    cv2.imshow('img',closing_w )

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        crop=closing_w[y+1:y+h,x+1:x+w]
        break

cap.release()
cv2.destroyAllWindows()
