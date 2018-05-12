import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image
import cv2
import math
from scipy import misc
import os
def zmain(loc):
	inLoc = os.path.basename(loc)
	# import Image

	### IMAGE INPUT ###
	#img_in = raw_input("Enter file destination:")
	img_in =loc
	img = Image.open(img_in)
	print "Original Image Location: ", img_in
	print "Original Image Dimensions: ", img.size

	print ""
	print "ENCRYPTION STARTS"
	start_time = time.time()
	print ""
	### IMAGE To BLACK & WHITE ###
	print "Converting to B&W"
	imgG = img.convert('1')
	# plt.imshow(imgG)
	# plt.show()

	WIDTH, HEIGHT = imgG.size

	# a = [[0 for x in range(w)] for y in range(h)]

	# IMAGE DATA IN 2D ARRAY #
	print "Image to 2D"
	data = list(imgG.getdata())
	data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]


	# FOR 255 TO 1#
	for y in range(HEIGHT):
	    for x in range(WIDTH):
		if data[y][x] == 255:
		    data[y][x] = 1



	# imgB = mpimg.imread(imgG, '1')

	# px = data[290][129]
	# print px
	data1 = np.array(data) # in np array for easy calc

	### ZIGZAG SCAN ###

	def zigz(x):
	    print "Zig-Zag Scan (2D-1D)"
	    # print "Image Dimensions before Zig-Zag Scan:", (x.shape)  # confirms 2d array
	    row, col = x.shape  # dimensions in variable

	    if row != col:
		print ('ZIGZAG Conversion fails!! Must be a square matrix!!')
		return

	    i = 0
	    j = 0

	    #upper triangle
	    zz = np.array([x[i][j]])
	    # print "ARRAY"
	    # print zz

	    inf = 1
	    while inf == 1:
		j = j+1
		zz = np.append(zz, [x[i][j]])

		while (j != 0):
		    i = i+1
		    j = j-1
		    zz = np.append(zz, [x[i][j]])

		i = i+1

		if (i > row-1):
		    i = i-1
		    break

		zz = np.append(zz, [x[i][j]])

		while(i != 0):
		    i = i-1
		    j = j+1
		    zz = np.append(zz, [x[i][j]])

	    # print ("hello")

	    #lower triangle
	    infy = 1
	    while infy == 1:
		# print ("hello")

		j = j + 1
		zz = np.append(zz, [x[i][j]])

		while (j != (col - 1)):
		    i = i - 1
		    j = j + 1
		    zz = np.append(zz, [x[i][j]])

		i = i + 1

		if (i > (row -1 )):
		    i = i - 1
		    break

		zz = np.append(zz, [x[i][j]])

		while (i != (row - 1)):
		    i = i + 1
		    j = j - 1
		    zz = np.append(zz, [x[i][j]])
	    return zz


	# Main Zigzag Function Call
	zz = zigz(data1)
	print "1D Zig-Zag Size: ", zz.size
	# print zz[0], zz[1], zz[2], zz[3], zz[4], zz[5], zz[6]



	### SHARE GENERATION ###
	print "Share Generation (S1 & S2)"
	yw = np.empty(0)
	yb = np.empty(0)

	for i in range(zz.size):
	    if zz[i] == 1:
		yw = np.append(yw, [i])
	    else:
		yb = np.append(yb, [i])


	lw = yw.size
	lb = yb.size
	print "White & Black size:", lw, lb

	s1 = np.zeros(zz.size, dtype=np.int8)
	s2 = np.zeros(zz.size, dtype=np.int8)


	# White Pixels Process #
	for i in range(0 , lw//4):
	    p = np.array([1, 1, 0, 0])
	    # print p[3]
	    for j in range(0, 4):
		if 4*i-(4-j) <= lw:
		    ind = np.int8(yw[4*i-(4-j)])

		    # np.insert(s1, index, p[j])
		    # np.insert(s1, index, p[j])

		    # s1[ yw[4*i-(4-j)] ] = p[j]
		    s1[ind] = p[j]
		    s2[ind] = p[j]


	# Black Pixels Process #
	for i in range(0, lb//4):
	    p1 = np.array([1, 1, 0, 0])
	    p2 = np.array([0, 0, 1, 1])

	    for j in range(0, 4):
		if 4*i-(4-j) <= lb:
		    ind = np.int8(yb[4*i-(4-j)])
		    # np.insert(s1, index, p1[j])
		    # np.insert(s2, index, p2[j])
		    # a=2
		    s1[ind] = p1[j]
		    s2[ind] = p2[j]

	print "Share 1: ", s1
	print "Sahre 2: ", s2

	print ""
	print "ENCRYPTION ENDS"
	zetime=time.time() - start_time	
	#IMAGE ENCRYPTION ENDS#

	### IMAGE RECOVERY ###
	print ""
	print ""
	start_time = time.time()
	print "DECRYPTION STARTS"
	r = np.bitwise_or(s1, s2)
	print r
	# print np.unique(r) #confirms 0 n 1 value in array



	# INVERSE ZIGZAG #
	def invZigzag(x):
	    print "Inverse Zig-Zag to convert 1D to 2D"
	  #   print (x.shape)  # confirms 2d array
	    row = int(math.sqrt(len(x)))  # dimensions in variable
	    col=row
	    if row != col:
		print ('ZIGZAG Conversion fails!! Must be a square matrix!!')
		return

	    i = 0
	    j = 0
	    temp=0
	    y = np.zeros([row, row], dtype=int)
	    # upper triangle
	    y[i][j]=x[temp]
		# print "ARRAY"
	    print y[0][0]

	    inf = 1
	    while inf == 1:
		j = j + 1
		temp=temp+1
		y[i][j] = x[temp]

		while (j != 0):
		    i = i + 1
		    j = j - 1
		    temp=temp+1
		    y[i][j] = x[temp]

		i = i + 1

		if (i > row - 1):
		    i = i - 1
		    break

		temp = temp + 1
		y[i][j] = x[temp]

		while (i != 0):
		    i = i - 1
		    j = j + 1
		    temp = temp + 1
		    y[i][j] = x[temp]

		# print ("hello")

		# lower triangle
	    infy = 1
	    while infy == 1:
		    # print ("hello")

		j = j + 1
		temp = temp + 1
		y[i][j] = x[temp]
		while (j != (col - 1)):
		    i = i - 1
		    j = j + 1
		    temp = temp + 1
		    y[i][j] = x[temp]

		i = i + 1

		if (i > (row - 1)):
		    i = i - 1
		    break

		temp = temp + 1
		y[i][j] = x[temp]

		while (i != (row - 1)):
		    i = i + 1
		    j = j - 1
		    temp = temp + 1
		    y[i][j] = x[temp]

	    return y
	def psnr(img1, img2):
		mse = np.mean( (img1 - img2) ** 2 )
		if mse == 0:
			return 100
		PIXEL_MAX = 255.0
		return 10 * math.log10(PIXEL_MAX / math.sqrt(mse))

	    # img =Image.fromarray(y)
	    #img.save("output.png")
	    # print(np.matrix(y))

	inz = invZigzag(zz)
	print "Image Dimensions after Inverse Zig-Zag: ", inz.shape
	print "IMAGE AFTER DECRYPTION"
	zdtime=time.time() - start_time
	plt.imshow(inz)

	plt.savefig("/home/ubuntu/Desktop/major/static/zd.jpg")
	image=cv2.imread('/home/ubuntu/Desktop/major/static/yd.jpg')
 	zpsnr = psnr(img,image)

	a = ['/static/'+inLoc, 'n', '/static/zd.jpg', zetime, zdtime,zpsnr]
    	return a

#zmain()

