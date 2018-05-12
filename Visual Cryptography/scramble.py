#!/usr/bin/python

from PIL import Image
import time
import sys
import scipy
import math
from scipy.misc import *
import cv2
import numpy as np
import os
import random

def openImage(img_l):
    img_loc = img_l
    #img_loc = "/Users/yash/PycharmProjects/Major/test2.png"
    img = Image.open(img_loc)
    #print "Original Image Location: ", img_loc
    #print "Original Image Dimensions: ", img.size
    return img

def openImage2():
    return Image.open("/home/ubuntu/Desktop/major/static/e.png")

def operation():
     return sys.argv[1]

def seed(img):
    random.seed(img.size)

def getPixels(img):
    w, h = img.size
    pxs = []
    for x in range(w):
        for y in range(h):
            pxs.append(img.getpixel((x, y)))
    return pxs

def scrambledIndex(pxs):
    idx = range(len(pxs))
    random.shuffle(idx)
    return idx

def scramblePixels(img):
    seed(img)
    pxs = getPixels(img)
    idx = scrambledIndex(pxs)
    out = []
    for i in idx:
        out.append(pxs[i])
    return out

def unScramblePixels(img):
    seed(img)
    pxs = getPixels(img)
    idx = scrambledIndex(pxs)
    out = range(len(pxs))
    cur = 0
    for i in idx:
        out[i] = pxs[cur]
        cur += 1
    return out

def storePixels(name, size, pxs):
    outImg = Image.new("RGB", size)
    w, h = size
    pxIter = iter(pxs)
    for x in range(w):
        for y in range(h):
            outImg.putpixel((x, y), pxIter.next())
    outImg.save(name)

def mainF(img_l):
	inLoc = os.path.basename(img_l)
	#print(os.path.basename(your_path))
	img = openImage(img_l)
	# if operation() == "scramble":
	#print "Image Scrambling Starts"
	start_time = time.time()
	pxs = scramblePixels(img)
	storePixels("/home/ubuntu/Desktop/major/static/e.png", img.size, pxs)
	setime=time.time() - start_time
	#print "Encrypted Image Saved at: /home/ubuntu/Desktop/major/static/e.PNG"
	# elif operation() == "unscramble":
	img2=openImage2()

	start_time = time.time()
	pxs = unScramblePixels(img2)
	storePixels("/home/ubuntu/Desktop/major/static/d.png", img2.size, pxs)
	sdtime=time.time() - start_time
	#print "Decrypted Image Saved at: /home/ubuntu/Desktop/major/static/d.PNG"
	def psnr(img1, img2):
		mse=np.mean((img1-img2)**2)
		if mse == 0:
			return 100
		PIXEL_MAX = 255.0
		return 10 * math.log10(PIXEL_MAX / math.sqrt(mse))
	image11=imread('/home/ubuntu/Desktop/major/static/d.png')
	img1=imread(img_l) 
	ssnr=psnr(img1,image11)
	
	# else:
	#     sys.exit("Unsupported operation: " + operation())
	# print "done"
	a = ['/static/'+inLoc, '/static/e.png', '/static/d.png',setime,sdtime,ssnr]	
	return a

# if _name_ == "_main_":
#mainF('/home/ubuntu/Desktop/test.jpg')
