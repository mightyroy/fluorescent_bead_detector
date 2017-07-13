import sys
import numpy as np
import cv2
import copy
import pandas
import os
import csv

def writecsv(data):
	with open('output.csv', 'w') as fp:
	    a = csv.writer(fp, delimiter=',') 
	    #data = [['Me', 'You'],
		#         ['293', '219'],
	    #         ['54', '13']]
	    a.writerows(data)


def rgbtohsv(r,g,b):
	# rgb to hue
	r, g, b = r/255.0, g/255.0, b/255.0
	mx = max(r, g, b)
	mn = min(r, g, b)
	df = mx-mn
	if mx == mn:
		h = 0
	elif mx == r:
		h = (60 * ((g-b)/df) + 360) % 360
	elif mx == g:
		h = (60 * ((b-r)/df) + 120) % 360
	elif mx == b:
		h = (60 * ((r-g)/df) + 240) % 360
	if mx == 0:
		s = 0
	else:
		s = df/mx
	v = mx
	return h, s, v

def processImage(imgFilename,data):
	MIN_BEAD_AREA = 100
	CUT_OFF_AREA = 25000
	DARKNESS_THRESH50D = 20 #out of 255
	GREENBOX_SIZE = 30

	#get image by filename
	# img =cv2.imread(str(sys.argv[1]))
	img = cv2.imread(imgFilename)
	img2 = copy.copy(img)

	# Find size of img
	sz = img.shape

	#get grayscale img
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_gray = img_gray

	#maximum among the 3 color channels
	img_max = np.amax(img,axis=2)
	img_blue = img[:,:,0] 
	img_green = img[:,:,1] 
	img_red = img[:,:,2] 


	#pick out colorful regions
	(B, G, R) = cv2.split(img)
	rg = np.absolute(R - G) # compute rg = R - G
	yb = np.clip(0.5 * (R + G) - B,0,255) # compute yb = 0.5 * (R + G) - B
	addgreen = np.clip(G - G.mean(),0,255)
	addred = np.clip(R - R.mean(),0,255)
	addblue = np.clip(B - B.mean(),0,255)
	#stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2)) # compute the mean and standard deviation of both `rg` and `yb`,
	#meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
	newmetricimage = np.array(rg + yb  , np.uint8) 
	#newmetricimage = np.array(rg + yb   , np.uint8) 
	np.clip(newmetricimage, 0, 255)
	#newmetricimage2 = cv2.cvtColor(newmetricimage, cv2.color_bgr2gray)

	#picks out bead centers for image#
	#ret, thresh = cv2.threshold(newmetricimage,cv2.DARKNESS_THRESHOLD,255,cv2.THRESH_BINARY)
	ret, thresh = cv2.threshold(newmetricimage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#ret, thresh = cv2.threshold(thresh,15,255,cv2.THRESH_BINARY)
	#thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,-10)
	#thresh = cv2.adaptiveThreshold(newmetricimage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,- 25)

	#noise removal
	kernel = np.ones((20,20),np.uint8)
	noiselessimg = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(noiselessimg,2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.43*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	sure_fg_copy = copy.deepcopy(sure_fg)

	#get contours
	contours, hierarchy = cv2.findContours( sure_fg_copy,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	#process individual contours
	allColors = []
	hist = []
	beadcount = 0
	for i, contour in enumerate(contours):
		if cv2.contourArea(contour) > MIN_BEAD_AREA :
			area = cv2.contourArea(contour)

			if area > CUT_OFF_AREA:
				#For overlapping beads
				x,y,w,h = cv2.boundingRect(contour)
				cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
				#do nothing elses
		
			else:       
			 M = cv2.moments(contour)
			 #calculate centroid
			 #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
			 cx = int(M['m10']/M['m00'])
			 cy = int(M['m01']/M['m00'])

			 PADDING = 30 
			 
			 #check if bead is not at edge of image
			 if cx > PADDING and cx < (img.shape[1]-PADDING) and cy > PADDING and cy< (img.shape[0]-PADDING):
				 cropsquarecoords = [cx,cy] #[2,]
				 #print("x= %i, y= %i", cx, cy)
				 cv2.rectangle(img, (cx-GREENBOX_SIZE, cy-GREENBOX_SIZE), (cx+GREENBOX_SIZE, cy+GREENBOX_SIZE), (0, 255, 0), 6)
				 #cv2.rectangle(img2_alignedboxed, (cx-7, cy-7), (cx+7, cy+7), (0, 255, 0), 2)

				 imgcropsquare = img2[(cy -GREENBOX_SIZE ):(cy + GREENBOX_SIZE),(cx -GREENBOX_SIZE ):(cx + GREENBOX_SIZE) ,:]
				 #wholeBeadCrop = img2[(cy -BEAD_RADIUS ):(cy + BEAD_RADIUS),(cx -BEAD_RADIUS ):(cx + BEAD_RADIUS),: ]
				 
				 mean = cv2.mean(imgcropsquare) #[3,]
			 
				 rgb = [mean[2],mean[1],mean[0]]
				 #convert rgb to hue
				 h,s,v = rgbtohsv(mean[2],mean[1],mean[0])

				 # HSP Color Model
				 #sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
				 l = np.sqrt(0.299*mean[2]**2 + 0.587*mean[1]**2 + 0.114*mean[0]**2)
				 allColors.append([h,rgb,l,cropsquarecoords])

				 #Hue,Saturation,Value, R, G, B, luminosity, image_name
				 data.append([h,s,v,mean[2],mean[1],mean[0],l,imgFilename])


				 
			
				 beadcount += 1

	if beadcount == 0:
		print "beadcount in %s is 0 !" % imgFilename
		#os.remove("./" + imgFilename)

	else:	
		print "%s bead count = %i " % (imgFilename ,beadcount)
		colorsDataFrame = pandas.DataFrame(allColors)
		colorsDataFrame.columns = ['hue','rgb','luminance', 'coords']

		sortedDF = colorsDataFrame.sort_values(by=['hue'],ascending=[True])


		#show image
		resizedfgcopy = cv2.resize(thresh,(500,400))
		cv2.imshow('All carbon lib foregrounds',resizedfgcopy)
		cv2.moveWindow('All carbon lib foregrounds',0,400)
		resizedimg = cv2.resize(sure_fg,(500,400))
		cv2.imshow('All carbon lib',resizedimg)
		resizedimg = cv2.resize(img,(500,400))
		cv2.imshow(imgFilename,resizedimg)
		cv2.moveWindow(imgFilename,500,0)
		colorfulmetricimg = cv2.resize(newmetricimage, (500,400))
		cv2.imshow('colorfulness',colorfulmetricimg)
		cv2.moveWindow('colorfulness',500,450)
		k = cv2.waitKey(0)
		# print (k)
  #  		if k == 120:
  #  			print (imgFilename + " saved")
  #  			os.rename("./" + imgFilename, "./selected/" + imgFilename)
  #  			cv2.destroyAllWindows()
  #  		else:
		# 	os.remove("./" + imgFilename)
		cv2.destroyAllWindows()


print "Place imagefiles in same directory as this file."
path = "./"
dirs = os.listdir(path)

data = []
data.append(["Hue","Saturation","Value", "R", "G", "B", "luminosity", "image_name"])

print "Press 'x' to save image"
for file in dirs:
	if file.endswith('.JPG'):
		processImage(file,data)


writecsv(data)




