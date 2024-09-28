# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 05:02:42 2021

@author: khosro
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt

import keyboard

#img_1 = cv2.imread('crop.jpg')
#
#gray_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#
#print(np.shape(gray_img))
#
#cv2.imshow('crop', gray_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#
#edges = edges / edges.max() #normalizes data in range 0 - 255
#edges = 255 * edges
#edges = edges.astype(np.uint8)
#
#plt.subplots(1,figsize=(10, 10))
#plt.subplot(111),plt.imshow(edges,cmap='gray'),plt.title('canny edges')
#
#lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)
#
#black=np.zeros(np.shape(gray_img))
#
#for rho, theta in lines[0]:
#    a = np.cos(theta)
#    b = np.sin(theta)
#    x0 = a * rho
#    y0 = b * rho
#    x1 = int(x0 + 1000 * (-b))
#    y1 = int(y0 + 1000 * (a))
#    x2 = int(x0 - 1000 * (-b))
#    y2 = int(y0 - 1000 * (a))
#    cv2.line(gray_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#    
#    
#cv2.imshow('black',gray_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#%%

img = cv2.imread('crop.jpg')
#print(np.shape(img))
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('crop', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray=img
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)


low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.subplots(1,figsize=(10, 10))
plt.subplot(111),plt.imshow(edges,cmap='gray'),plt.title('canny edges')
rho = 3  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 30  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
n=0
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)
        n=n+1


# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

cv2.imshow('line_image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('line_image.jpg',img)

print(n)





#%%

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
            
        closing_w=img
        
    else:
        
        skin_mask_w=human_skin_detector(img,minr,ming,minb,maxr,maxg,maxb,var)
        skin_mask_w = skin_mask_w.astype(np.uint8)
        skin_mask_w*=255
        kernel = np.ones((11,11),np.uint8)
        mask_w = cv2.morphologyEx(skin_mask_w, cv2.MORPH_OPEN,kernel)
        
        kernel_2 = np.ones((3,3),np.uint8)
        closing_w = cv2.morphologyEx(mask_w , cv2.MORPH_CLOSE, kernel_2)
        
        contours, hierarchy = cv2.findContours(closing_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        print(n>7)
        #closing_w=img
        
    cv2.imshow('img',closing_w )

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        crop=closing_w[y+1:y+h,x+1:x+w]
        break

cap.release()
cv2.destroyAllWindows()

