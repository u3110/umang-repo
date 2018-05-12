import time
import random
import halftones
import cv2
import scipy
import math
from scipy.misc import *
import numpy as np
def GenerateRandDots(height,width):
    m=np.ndarray(shape=(6,1,4))
    print(random.randint(0, 5))
    key=[];
    curRows=[];
    m[0] =[0,0,1,1]
    m[1,0] = [1,1,0,0]
    m[2,0] = [1,0,1,0]
    m[3,0] = [0,1,0,1]
    m[4,0] = [1,0,0,1]
    m[5,0] = [0,1,1,0]
    for i in range(int(height/2)):
        for j in range(int(width/2)):
            curRows.append( m[random.randint(0, 5)])
        p=np.array(curRows)
        shp=np.shape(p)
        arr=np.reshape(p,(1,shp[0]*shp[1]*shp[2]))
        key.append(arr)
        curRows=[]
    key=np.array(key)
    key=np.reshape(key ,(height,width))
    return key
def EncryptandSave(image,key):
    r = halftones.halftone.error_diffusion_jarvis(image[0:, 0:, 0])
    g = halftones.halftone.error_diffusion_jarvis(image[0:, 0:, 1])
    b = halftones.halftone.error_diffusion_jarvis(image[0:, 0:, 2])
    encryptedR = np.logical_xor(r, key);
    encryptedG = np.logical_xor(g, key);
    encryptedB = np.logical_xor(b, key);
    encryptedImage = np.zeros([rows, column, 3])
    encryptedImage[:, :, 0] = encryptedR
    
    #imshow(encryptedImage)
    encryptedImage[0:, 0:, 1] = encryptedG
    #imshow(encryptedImage)
    encryptedImage[0:, 0:, 2] = encryptedB
    #imshow(encryptedImage)
    imsave("encryptedimage.jpg",encryptedImage)

    return encryptedImage
def DecryptandSave(image,key):
    encryptedR=image[:,:,0]
    encryptedG=image[:,:,1]
    encryptedB=image[:,:,2]
    HalftonedecryptedR = np.logical_xor(encryptedR, key);
    HalftonedecryptedG = np.logical_xor(encryptedG, key);
    HalftonedecryptedB = np.logical_xor(encryptedB, key);

    invHalfdecryptedR = halftones.inverse_halftone.inverse_fbih(HalftonedecryptedR)
    invHalfdecryptedG = halftones.inverse_halftone.inverse_fbih(HalftonedecryptedG)
    invHalfdecryptedB = halftones.inverse_halftone.inverse_fbih(HalftonedecryptedB)

    DecryptedImage = np.zeros([rows, column, 3])
    DecryptedImage[:, :, 0] = invHalfdecryptedR
    DecryptedImage[0:, 0:, 1] = invHalfdecryptedG
    DecryptedImage[0:, 0:, 2] = invHalfdecryptedB
    imsave("decryptedimage.jpg", DecryptedImage)
	
   # imshow(DecryptedImage)
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ymain(loc):
	#path="/home/umang/Desktop/test.jpg"
	path=loc
	image=imread(path)
	orignalimage=image
	shape=np.shape(image);
	r=shape[0]
	c=shape[1]
	if(r%2!=0):
		image = cv2.copyMakeBorder(image,0,1,0,0,cv2.BORDER_REPLICATE)
	if(c%2!=0):
    		image = cv2.copyMakeBorder(image, 0, 0, 0, 1, cv2.BORDER_REPLICATE)


	shap=np.shape(image)

	rows=shap[0]
	column=shap[1]
	share_1=GenerateRandDots(rows,column)
	imsave("share-1.jpg",share_1)

	start_time = time.time()
	enc=EncryptandSave(image,share_1)
	print("EncryptionTime--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	DecryptandSave(enc,share_1)

	print("DecryptionTime--- %s seconds ---" % (time.time() - start_time))

	image=imread('decryptedimage.jpg')
	snr = psnr(orignalimage,image)
	print("Peak Signal To Noise Ratio:",snr)



