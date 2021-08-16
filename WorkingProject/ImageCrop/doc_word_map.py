#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:00:32 2021

@author: Piyush
"""


import cv2
import numpy as np
import math
import os 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import statistics


##############Modify input file path accordingly#############
filepath = "/home/XXXXX/IMG-20180710-WA0013.jpg"
filepath_line_res = "/home/XXXX/sample_test_line.jpg"
dirpath_cropline = "/home/XXXX/sample_lines"


class Stack:

    #Constructor creates a list
    def __init__(self):
        self.stack = list()

    #Adding elements to stack
    def push(self,data):
        self.stack.append(data)
        return True


    #Removing last element from the stack
    def pop(self):
        if len(self.stack)<=0:
            return ("Stack Empty!")
        return self.stack.pop()
        
    #Getting the size of the stack
    def size(self):
        return len(self.stack)



def rem_overlapped(borders,area):
    #check overlapped rec
    res_borderd = []
    for k in range(2):   
        if len(res_borderd) > 0 :
            borders = res_borderd
            res_borderd = []
      
        for i,c in enumerate(borders):
        
            x11,y11,x12,y12,subarea1,ismerg = c['x1'], c['y1'], c['x2'], c['y2'],c['sum'],c['dup']
        
            if subarea1 > (0.03 * area):
                continue
            
            #if ismerg == 1:
            #    continue
            if (y12-y11) > 17:
                continue
            
            xl = x11
            yl = y11
            xh = x12
            yh = y12
            
            for j,d in enumerate(borders):
                #if j <= i:
                #    continue
                
                x21,y21,x22,y22,subarea2,ismerg2 = d['x1'], d['y1'], d['x2'], d['y2'],d['sum'],d['dup']    
                
                if subarea2 > (0.03 * area):
                    continue
                
                #if ismerg2 ==  1:
                #    continue
                
                if (y22-y21) > 17:
                    continue
                
                if (xh < x21) or (xl > x22) or (yh < y21) or (yl > y22):
                    continue
                else:
                    xl = min(xl,x21)
                    yl = min(yl,y21)
                    xh = max(xh,x22)
                    yh = max(yh,y22)
                    borders[j]['dup'] = 1;
            
            if (yh-yl) < 17:                
                res_borderd.append({'x1':xl,'y1':yl,'x2':xh,'y2': yh, 'sum': ((xh-xl)*(yh-yl)), 'dup':0})
         
    return res_borderd


def rem_neighbour(borders):
    #check neighbour rect
    res_borderd = []    
   
    for i,c in enumerate(borders):
    
        x11,y11,x12,y12,subarea1,ismerg = c['x1'], c['y1'], c['x2'], c['y2'],c['sum'],c['dup']
    
        #if subarea1 > (0.1 * area):
        #    continue
        
        if ismerg == 1:
            continue
        
        xl = x11
        yl = y11
        xh = x12
        yh = y12
        borders[i]['dup'] = 1;
        
        for j,d in enumerate(borders):
            if j <= i:
                continue
            
            x21,y21,x22,y22,subarea2,ismerg2 = d['x1'], d['y1'], d['x2'], d['y2'],d['sum'],d['dup']    
            
            #if subarea2 > (0.1 * area):
            #    continue
            
            if ismerg2 ==  1:
                continue
            
            if (x21 - xh > 10) or (xl - x22 > 10) or (y21 - yh > 5) or (yl - y22 > 5):
                continue
            else:
                xl = min(xl,x21)
                yl = min(yl,y21)
                xh = max(xh,x22)
                yh = max(yh,y22)
                borders[j]['dup'] = 1;

        if  (xh-xl)*(yh-yl) < 50 :                    
            res_borderd.append({'x1':xl,'y1':yl,'x2':xh,'y2': yh, 'sum': ((xh-xl)*(yh-yl)), 'dup':0})
                
    return res_borderd


def merge_rec(border):
   
    myStack = Stack()
    myQueue = list()
    res_borderd
    for i,c in enumerate(borders):
        x11,y11,x12,y12,subarea1,ismerg = c['x1'], c['y1'], c['x2'], c['y2'],c['sum'],c['dup']
       
        #if ismerg == 1:
        #    continue
        
       xl = x11
       yl = y11
       xh = x12
       yh = y12
        
       for j,d in enumerate(borders):
           #if j <= i:
           #    continue
            
           x21,y21,x22,y22,subarea2,ismerg2 = d['x1'], d['y1'], d['x2'], d['y2'],d['sum'],d['dup']    
            
            #if ismerg2 ==  1:
            #    continue
           
            
           if (xh < x21) or (xl > x22) or (yh < y21) or (yl > y22):
               if (((y21 > yl and y21 < yh) or (yh > y21 and yh < y22)) and (abs(xh-x21) < 3)):
                   #xl = min(xl,x21)
                   #yl = min(yl,y21)
                   #xh = max(xh,x22)
                   #yh = max(yh,y22)
                   borders[j]['dup'] = 1;
                   myStack.push({'x':xl,'y':yl})
                   myStack.push({'x':xh,'y':yl})
                   myStack.push({'x':x22,'y':y21})
                   myQueue.append({'x':xl,'y':})
        

    res_borderd.append({'x1':xl,'y1':yl,'x2':xh,'y2': yh, 'sum': ((xh-xl)*(yh-yl)), 'dup':0})
    pts = np.array([[val[0],val[1]-ofst],[val[2],val[3]-ofst],[val[2],val[3]+ofst],[val[0],val[1]+ofst]])                      
         
         
         

def find_duplicates(values):
    output = []    
    maxid = max(values)
    i = 0;
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value == maxid :
            output.append(i)
        i= i+1
    return output

def draw_contours(file,file_merge):
    img = cv2.imread(file,0)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    ##################
    borders = []
    area = edges.shape[0] * edges.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        #if w * h > 0.0005 * area: #if w * h > 0.5 * area:
        borders.append({'i': i,
                           'x1':x,
                           'y1':y,
                           'x2':x + w - 1,
                           'y2': y + h - 1,
                           'sum': (w-1)*(h-1),
                           'dup' : 0})
        #cv2.rectangle(img, (x,y), (x+w-1,y+h-1), (0,255,0))
        
    res_borderd = rem_overlapped(borders,area)
    #final_borderd = rem_neighbour(res_borderd)
    #res_borderd = rem_overlapped(final_borderd,area)
    #final_borderd = rem_neighbour(res_borderd)

    #img1 = cv2.imread(file)
    img1 = Image.open(file)
    draw = ImageDraw.Draw(img1)    
    for i, c in enumerate(res_borderd):
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        draw.rectangle(this_crop, outline='red')
        cv2.imwrite(file_merge,img[c['y1']:c['y2'], c['x1']:c['x2']])
    #im.save(out_path)
    del draw
        
    #cv2.imwrite(file,img)    
    

def rotate_line(ang,img):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -ang, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH))


def crop_line_img(val,ofst,path):
    img = cv2.imread(filepath)
    pts = np.array([[val[0],val[1]-ofst],[val[2],val[3]-ofst],[val[2],val[3]+ofst],[val[0],val[1]+ofst]])        
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()        
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg + dst
    
    cv2.imwrite(path,dst2)    


def check_rotation(path):
    img = cv2.imread(path,0)    
    print(path)   
    th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    histr = cv2.calcHist([th1],[0],None,[256],[0,256]) 
    print(histr[0])

    # show the plotting graph of an image 
    #plt.plot(histr) 
    #plt.show()
    cv2.imwrite(path,th1)    
    
    
def draw_crop(values,ig):
    ofst = 7 # rect of height 3
    #img = cv2.imread(filepath)

    #name,ext = os.path.splitext(dirpath_cropline)    
    if not os.path.exists(dirpath_cropline):
        os.makedirs(dirpath_cropline) 
    filename = os.path.basename(filepath)
    
    i = 0
    for item in values:
        x1 = item[0]
        y1 = item[1]
        x2 = item[2]
        y2 = item[3]+3  #line height
        a = item[4]
        #img_line = dirpath_cropline + '/'+filename+'_'+ str(i) +'.png'
        img_line_1 = dirpath_cropline + '/'+filename+'_'+ str(i) +'_temp.png'
        i = i+1
        lnagl = a
        if int(a) == 0 :
            print("Angle of rotation",90-int(a))
            a = 90 - int(a)
        elif int(a) != 90 :
            print("Angle of rotation",90-int(a))
            a = 90 - int(a)
        else:
            print("No rotation needed")
            
        if  lnagl > 75 :
            #print("Image is H position")
            ang_line = 'H'
        else:
            #print("Image is V position")
            ang_line = 'V'            
            
        if ang_line == ig:
            print("Ignore line: "+ang_line)
            continue
        
        crop_line_img(item,ofst,img_line_1)
        
        #check_rotation(img_line_1)
        draw_contours(img_line_1)
        
        #cv2.imshow("same size" , res)         
        #cv2.waitKey(0)
        
        #img_rt = rotate_line(a,img_line)
        #cv2.imwrite(imgname,imh_rt)
            
##################Image Tilt : HoughLine Method
img = cv2.imread(filepath)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,50,150,apertureSize = 3)
#minLineLength=img.shape[1]-300
#lines = cv2.HoughLinesP(image=edges,rho=0.02,theta=np.pi/500, threshold=10,lines=np.array([]), minLineLength=minLineLength,maxLineGap=100)

# This returns an array of r and theta values 
lines = cv2.HoughLines(edges,1,np.pi/180, 200) 

#a,b,c = lines.shape
#for i in range(a):
#    cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

avg_angl = 0
values = []
angles = []
for item in lines: 
    r,theta = item[0]
    a = np.cos(theta) 
    b = np.sin(theta) 
    x0 = a*r 
    y0 = b*r 
    x1 = int(x0 + 1000*(-b)) 
    y1 = int(y0 + 1000*(a)) 
    x2 = int(x0 - 1000*(-b)) 
    y2 = int(y0 - 1000*(a)) 
    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) 
    dev_ang = math.degrees(theta)
    if(dev_ang > 135) :
        dev_ang = 180 - dev_ang 
    if x1<0:
        x1 = 0
    if x2<0:
        x2 = 0
        
    print(dev_ang, x1,y1,x2,y2)
    avg_angl = avg_angl + dev_ang
    item = []
    item.append(x1)
    item.append(y1)
    item.append(x2)
    item.append(y2)
    item.append(int(dev_ang))
    values.append(item)
    angles.append(int(dev_ang))
    #values.append(x1,y1,x2,y2,int(dev_ang))
    
    #x1,y1,x2,y2,
#x = np.random.random_integers(1, 100, 5)
stdev = statistics.stdev(angles)
print("Std.Deviation ",stdev)
counts, bins, bars = plt.hist(angles, bins=20)
plt.xlabel('Angle')
plt.ylabel('No of times')
plt.show()

#dup = merge_duplicates(counts,bins)   
#if len(dup) == 1 :

maxidx = counts.argmax(axis=0)
if int(bins[maxidx]) == 0 :
    a = 90-int(bins[maxidx])
    print("Angle of rotation",90-int(bins[maxidx]))
elif int(bins[maxidx]) != 90 :
    a = 90-int(bins[maxidx+1])
    print("Angle of rotation",90-int(bins[maxidx+1]))
else:
    a = 0
    print("No rotation needed")
    
if  bins[maxidx] > 75 :
    print("Image is H position")
    ignore_line = 'V'
else:
    print("Image is V position")
    ignore_line = 'H'


    
#rt_img = rotate_line(a,img)
ret = draw_crop(values,ignore_line)
        
#rows,cols = img.shape
#M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_angle,1)
#dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imwrite(filepath_line_res, img)







