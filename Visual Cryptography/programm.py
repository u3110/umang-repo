#!/usr/bin/env python
import numpy as np
import RSA as rsa
import cv2
import sys
import re
import Elgamal as elgml
from scipy.misc import *

import time
import math
import os
#import pandas as pd

img1 = cv2.imread("/home/ubuntu/Desktop/test3.jpeg", 0)
def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y
def closestNumber(n, m):
    # Find the quotient
    for i in range(0,n*m,m):
      if(i>n):
          return i-n



        # else n2 i
def slice(img):
    img=np.array(img)
    shp=np.shape(img)
    height=shp[0]
    width=shp[1]
    count=0;
    slic=np.zeros([9,int(height/3),int(width/3)])
    m=height/3
    for i in range(0,height,int(height/3)):
        for j in range(0, width,int(width/3)):
            p=img[i:i+int(height/3),j:j+int(width/3)]
            p=np.reshape(p,[int(height/3),int(width/3)])
            slic[count, :, :]=p
            count=count+1
    return slic
def converttobin(img):
    shp=np.shape(img)
    cnt=0;
    lst = []
    bstr=''
    temp=np.reshape(img,[9,shp[2]])
    for arr in temp:
        for num in arr:
            bstr=bstr+'{0:08b}'.format(int(num))
            cnt=cnt+8
        lst.append(bstr)
        bstr=''

    return lst
def divideimageblocks(img):

    remnd=len(img)%64
    for i in range(0,(64-remnd)):
        img=img+'2'

    block=np.zeros([int(len(img)/64),64])
    length=int(len(img) / 64)

    count=0
    cnt_arr=0
    itrt=0
    p=np.array(img[0:64])

    for i in range(0,length):
        for j in range(0,64):
            block[itrt,cnt_arr]=img[count]

            count=count+1
            cnt_arr=cnt_arr+1
        itrt=itrt+1
        cnt_arr=0


    return block
def permute(BinaryArrayforPermut):
    f = open("after permutation.txt", "w+")

    config = np.loadtxt("/home/ubuntu/Desktop/major/permut.txt", delimiter=' ')
    shp = np.shape(BinaryArrayforPermut)
    permutedArray = np.zeros([shp[0], shp[1]])
    k = 0
    l = 0

    for i in range(0, shp[0] - 1):
        for j in range(0, shp[1]):
            indx = int(config[k, l])
            indx = indx - 1
            x = BinaryArrayforPermut[i,indx]
            permutedArray[i, j] = int(x)
            l = l + 1
            if l == 8:
                k = k + 1
                l = 0
        k = 0
        l = 0
    permutedArray[shp[0] - 1, 0:64] = BinaryArrayforPermut[shp[0] - 1, 0:64]
    return permutedArray
def converttodecimal(arr):
    nonzero=0;
    shp = np.shape(arr)
    for num in arr[shp[0]-1,:]:
        if(num>1):
            nonzero=nonzero+1


    leng=shp[0]*shp[1]
    arr=np.reshape(arr,[1,leng])
    newarr=arr[0,0:leng-nonzero]

    k=0
    shp=np.shape(newarr)
    temp=np.zeros([1,int(shp[0]/8)])
    for i in range(0,shp[0]-1,8):
        n=arr[0,i:i+8]
        temp[0,k]=n.dot(2**np.arange(n.size)[::-1])
        k=k+1

    return temp
def encrypt(arr,key):
    shp=np.shape(arr)
    temp=np.zeros(shp[0],shp[1])
    for i in range(0, shp[0]):
        for j in range(0,shp[1]):
            return #temp[i,j]=
def imagepresteps(plainslice,height,width):
    print(np.shape(plainslice))
    binimage = converttobin(plainslice)
     #print(len(binimage[0]))

    length = int(len(binimage[0]) / 64) + 1
    BinaryArrayforPermut = np.zeros([9, length, 64])

    for i in range(0, 9):
        BinaryArrayforPermut[i, :, :] = divideimageblocks(binimage[i])

    shparr = np.shape(BinaryArrayforPermut)

    permutedarray = np.zeros([shparr[0], shparr[1], shparr[2]])

    for i in range(0, 9):
        permutedarray[i,:,:]=permute(BinaryArrayforPermut[i,:,:]);
    decimalearray = np.zeros([9, int(height / 3), int(width / 3)])
    for i in range(0, 9):
        temp = converttodecimal(permutedarray[i])
        decimalearray[i] = np.reshape(temp, [int(height / 3), int(width / 3)])

    return decimalearray
def rversepermutaion(BinaryArrayforPermut):

    config = np.loadtxt("/home/ubuntu/Desktop/major/permut.txt", delimiter=' ')
    shp = np.shape(BinaryArrayforPermut)
    permutedArray = np.zeros([shp[0], shp[1]])
    k = 0
    l = 0

    for i in range(0, shp[0] - 1):
        for j in range(0, shp[1]):
            indx = int(config[k, l])
            indx=indx-1
            x= BinaryArrayforPermut[i,j]
            permutedArray[i, indx]=int(x)
            l = l + 1
            if l == 8:
                k = k + 1
                l = 0
        k = 0
        l = 0

    permutedArray[shp[0] - 1, 0:64] = BinaryArrayforPermut[shp[0] - 1, 0:64]
    return permutedArray
def reconstructimage(imgarr):

    seg1=np.hstack((imgarr[0,:,:], imgarr[1,:,:]))

    seg1=np.hstack((seg1,imgarr[2,:,:]))

    seg2 = np.hstack((imgarr[3, :, :], imgarr[4, :, :]))

    seg2 = np.hstack((seg2, imgarr[5, :, :]))

    seg3 = np.hstack((imgarr[6, :, :], imgarr[7, :, :]))

    seg3 = np.hstack((seg3, imgarr[8, :, :]))


    seg4=np.vstack((seg1,seg2))
    seg5=np.vstack((seg4,seg3))
    imsave("static/rd.png",seg5)
    #imshow(seg5)
    return seg5

def ElgamalEncy(imgseg,key):
    shpseg=np.shape(imgseg)

    encysegmnt = np.empty((shpseg[0], shpseg[1]), dtype=object)

    for i in range(0,shpseg[0]):
        for j in range(0, shpseg[1]):
            encysegmnt[i,j]=elgml.encrypt(key,str(imgseg[i,j]))

    return encysegmnt
def RSAEncy(imgseg,n,e,blocksize):
    shpseg = np.shape(imgseg)

    encysegmnt = np.empty((shpseg[0], shpseg[1]), dtype=object)

    for i in range(0, shpseg[0]):
        for j in range(0, shpseg[1]):
            encysegmnt[i, j] = rsa.encrypt( str(imgseg[i, j]),n,e,blocksize)

    return encysegmnt
def ElgamalDecy(encyseg,key):
    shpseg = np.shape(encyseg)
    decysegmnt = np.zeros([shpseg[0], shpseg[1]])

    for i in range(0, shpseg[0]):
        for j in range(0, shpseg[1]):
            decysegmnt[i, j] = elgml.decrypt(key, encyseg[i, j])

    return decysegmnt
    encysegmnt = np.empty((shpseg[0], shpseg[1]), dtype=object)

    for i in range(0, shpseg[0]):
        for j in range(0, shpseg[1]):
            encysegmnt[i, j] = elgml.encrypt(key, str(imgseg[i, j]))

    return encysegmnt
def RSADecy(encyseg,n,e,blocksize,d):
    shpseg = np.shape(encyseg)
    decysegmnt = np.zeros([shpseg[0], shpseg[1]])
    #
    for i in range(0, shpseg[0]):
        for j in range(0, shpseg[1]):
             k= re.findall('\d+', rsa.decrypt(encyseg[i, j],n,d,blocksize))
             decysegmnt[i, j]=int(k[0])

    return decysegmnt
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX / math.sqrt(mse))



def rsamain(imgloc):
	inLoc = os.path.basename(imgloc)
	img= cv2.imread(imgloc, 0)
	print ("Original Image Dimensions: ", img.size)
	shp = np.shape(img)
	height = shp[0]
	width = shp[1]
	heightindex=closestNumber(height,3)
	widthindex=closestNumber(width,3)
	if(height%3!=0):
		img = cv2.copyMakeBorder(img,0,heightindex,0,0,cv2.BORDER_REPLICATE)
	if(width%3!=0):
		img = cv2.copyMakeBorder(img, 0, 0, 0, widthindex, cv2.BORDER_REPLICATE)
	shp = np.shape(img)


	height = shp[0]

	width = shp[1]
	orignalimg=img
	start_time = time.time()

	slic=slice(img)



	plainslice=np.reshape(slic,[9,1,int(height/3)*int(width/3)])

	decimalearray=imagepresteps(plainslice,height,width)
	elgmlkeys=elgml.generate_keys()
	elgmlpub=elgmlkeys['publicKey']
	elgmlpriv=elgmlkeys['privateKey']
	ShareImgs = np.empty((9, int(height/3),int(width/3)), dtype=object)
	(n, e, d) = rsa.newKey(10**100, 10**101, 50)

	for i in range(0,9):
		if(i%2!=0):
			ShareImgs[i, :, :]=ElgamalEncy(decimalearray[i],elgmlpub)
		else:
			ShareImgs[i, :, :] = RSAEncy(decimalearray[i], n,e,8)



	rsaetime=(time.time() - start_time)
	print(rsaetime)
	print("Decryption Started")
	start_time = time.time()

	decrypteddecimalarray = np.zeros([9, int(height/3),int(width/3)])

	for i in range(0,9):
		if(i%2!=0):
			decrypteddecimalarray[i,:,:] =ElgamalDecy(ShareImgs[i],elgmlpriv)
		else:
			decrypteddecimalarray[i, :, :] = RSADecy(ShareImgs[i], n,d,8,d)


	plainslice=np.reshape(decrypteddecimalarray,[9,1,int(height/3)*int(width/3)])

	imagepresteps(plainslice,height,width)
	binimage = converttobin(plainslice)
	# print(len(binimage[0]))

	length = int(len(binimage[0]) / 64) + 1
	BinaryArrayforPermut = np.zeros([9, length, 64])

	for i in range(0, 9):
		BinaryArrayforPermut[i, :, :] = divideimageblocks(binimage[i])

	shparr = np.shape(BinaryArrayforPermut)

	reversedpermutedarray = np.zeros([shparr[0], shparr[1], shparr[2]])
	for i in range(0, 9):
		reversedpermutedarray[i, :, :] = rversepermutaion(BinaryArrayforPermut[i])

	decimalearrayreversepermuted = np.zeros([9, int(height / 3), int(width / 3)])
	for i in range(0, 9):
		temp = converttodecimal(reversedpermutedarray[i])
		decimalearrayreversepermuted[i] = np.reshape(temp, [int(height / 3), int(width / 3)])


	recoveredimg=reconstructimage(decimalearrayreversepermuted)
	rsadtime=(time.time() - start_time)
	print(rsadtime)

	rsapsnr=(psnr(orignalimg,recoveredimg))
	
	
	a = ['/static/'+inLoc, '/static/re.png', '/static/rd.png',rsaetime,rsadtime,rsapsnr]
	rsa_out = open("output.csv",'w+')
	rsa_out.write('/static/'+inLoc+','+ '/static/share-1.jpg'+','+ '/static/rd.png'+','+str(rsaetime)+','+str(rsadtime)+','+str(rsapsnr))
	rsa_out.close()
	#my = pd.DataFrame(a)
	#my.to_csv("output.csv",index=False,header=False)
	return a


print(sys.argv[1])
rsamain(sys.argv[1])
