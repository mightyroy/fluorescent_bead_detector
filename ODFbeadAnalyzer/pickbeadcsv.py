#Instructions: OpenCV3 installation required
#Put 2 ODF images in same folder
#input in command line: python pickbead.py firstImageName.TIF secondImageName.TIF
#Data will be saved in .csv file

import sys
sys.path.append('/usr/local/Cellar/opencv/2.4.11_1/lib/python2.7/site-packages')
import numpy as np
import cv2
import copy
import csv

BEAD_AREA_BEFORE = 50
BEAD_AREA_AFTER = 50

#input in command line: python pickbead.py firstImage.TIF secondImage.TIF
if len(sys.argv) == 1:
	img =cv2.imread('YYYYbefore.TIF')
	img2 =  cv2.imread("YYYYafter.TIF")
else: 
	img =cv2.imread(str(sys.argv[1]))
	img2 =  cv2.imread(str(sys.argv[2]))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

####This part aligns images

# Find size of image1
sz = img.shape

# Define the motion model
warp_mode = 0 #warp_mode == cv2.MOTION_TRANSLATION

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == 3: #warp_mode == cv2.MOTION_HOMOGRAPHY 
	warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
	warp_matrix = np.eye(2, 3, dtype=np.float32)
 
# Specify the number of iterations.
number_of_iterations = 5000;

 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix. (only works in opencv3)
(cc, warp_matrix) = cv2.findTransformECC (img_gray,img2_gray,warp_matrix, warp_mode, criteria)

#make a copy of image2 for grey bounding box to show translation
img2boxed = img2.copy()
cv2.rectangle(img2boxed, (0,0), (img2boxed.shape[1]-2,img2boxed.shape[0]-2),(255,240,245),2)

if warp_mode == cv2.MOTION_HOMOGRAPHY :
	# Use warpPerspective for Homography 
	img2_aligned = cv2.warpPerspective (img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	
else :
	# Use warpAffine for Translation, Euclidean and Affine
	img2_aligned = cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	img2_alignedboxed = cv2.warpAffine(img2boxed, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) #boxed image2 to show translation
	
	
img2_aligned_gray = cv2.cvtColor(img2_aligned,cv2.COLOR_BGR2GRAY)
 
##### End of image alignment ######

img1prestine = img.copy()
img2alignedprestine = img2_aligned.copy()

####this part creates gray difference image####
img128first = img.copy()
img128second = img2_aligned.copy()
difference128 = img128first - img128second +128
resizedX = cv2.resize(difference128,(500,400))
cv2.imshow('difference image', resizedX)
cv2.moveWindow('difference image',500,400)
###end of grey image creation####

#####This part picks out bead centers for first image####
ret, thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
noiselessimg = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
#sure_bg = cv2.dilate(noiselessimg,kernel,iterations=3)

# Finding sure foreground area
#dist_transform = cv2.distanceTransform(noiselessimg,cv2.DIST_L2,5)
dist_transform = cv2.distanceTransform(noiselessimg,2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
sure_fg_copy = copy.deepcopy(sure_fg)
#unknown = cv2.subtract(sure_bg,sure_fg)

_, contours, hierarchy = cv2.findContours( sure_fg_copy,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#print("count =%i", len(contours))

beadcountbefore = 0
for i, contour in enumerate(contours):
	if cv2.contourArea(contour) > BEAD_AREA_BEFORE :
	   #area = cv2.contourArea(contour)
	   #print("area =%i", area)
	   M = cv2.moments(contour)
	   #calculate centroid
	   #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
	   cx = int(M['m10']/M['m00'])
	   cy = int(M['m01']/M['m00'])
	   #print("x= %i, y= %i", cx, cy)
	   cv2.rectangle(img, (cx-7, cy-7), (cx+7, cy+7), (0, 255, 0), 2)
	   cv2.rectangle(img2_alignedboxed, (cx-7, cy-7), (cx+7, cy+7), (0, 255, 0), 2)
	   beadcountbefore += 1


######End of bead center picking for first image #######

#####This part picks out bead centers for second image####
ret2, thresh2 = cv2.threshold(img2_aligned_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
noiselessimg2 = cv2.morphologyEx(thresh2,cv2.MORPH_OPEN,kernel, iterations = 2)

# Finding sure foreground area
#dist_transform = cv2.distanceTransform(noiselessimg,cv2.DIST_L2,5)
dist_transform2 = cv2.distanceTransform(noiselessimg2,2,5)
ret2, sure_fg2 = cv2.threshold(dist_transform2,0.7*dist_transform.max(),255,0)

sure_fg2 = np.uint8(sure_fg2)

_, contours2, hierarchy = cv2.findContours( sure_fg2,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#print("count =%i", len(contours2))

beadcountafter = 0
for contour in contours2:
	if cv2.contourArea(contour) > BEAD_AREA_AFTER :
	   #area = cv2.contourArea(contour)
	   #print("area =%i", area)
	   M = cv2.moments(contour)
	   #calculate centroid
	   #http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
	   cx = int(M['m10']/M['m00'])
	   cy = int(M['m01']/M['m00'])
	   #print("x= %i, y= %i", cx, cy)
	   cv2.rectangle(img2_alignedboxed, (cx-7, cy-7), (cx+7, cy+7), (0, 0, 255), 2)
	   beadcountafter += 1
######End of bead center picking for second image #######

##### this part checks for bounding box overlap before and after#####

mask = np.zeros(img.shape[:2], np.uint8)
#blank_image = np.zeros((0,0,3), np.uint8)

matchingBeadCount = 0

csvdata = ()
csvlabel = ('BeadNumBefore', 'R', 'G', 'B', 'BeadNumAfter', 'R', 'G', 'B', 'deltaR', 'deltaG', 'deltaB')
csvdata = csvdata + (csvlabel,)

length = len(csvdata[0])

for i, c1 in enumerate(contours):
	if cv2.contourArea(c1) > BEAD_AREA_BEFORE:
		M1 = cv2.moments(c1)
		c1x = int(M1['m10']/M1['m00'])
		c1y = int(M1['m01']/M1['m00'])
		for j, c2 in enumerate(contours2):
			if cv2.contourArea(c2) > BEAD_AREA_AFTER:
				M2 = cv2.moments(c2)
				c2x = int(M2['m10']/M2['m00'])
				c2y = int(M2['m01']/M2['m00'])
				dist = np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2)
				if dist < 20: 
					matchingBeadCount += 1

					
					
					img1cropsquare = img1prestine[(c1y -7 ):(c1y + 7),(c1x -7 ):(c1x + 7) ]
					mean1 = cv2.mean(img1cropsquare)
					
					img2cropsquare = img2alignedprestine[(c2y-7):(c2y+7), (c2x-7):(c2x+7)]
					mean2 = cv2.mean(img2cropsquare)

					deltaR = mean2[2] - mean1[2]
					deltaG = mean2[1] - mean1[1]
					deltaB = mean2[0] - mean1[0]
					
					#if i == 10 and j == 10:
					#	cv2.imshow('crop',img1cropsquare)

					#print "bead %i , %s" % (i, mean1)
					#print "bead %i , %s" % (i, mean2)

					print "[bead %i before] & [bead %i after]: R %.2f, G %.2f, B %.2f" % (i,j, deltaR, deltaG, deltaB )
					cv2.putText(img,str(i), (c1x , c1y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2) #number text for each bead
					cv2.putText(img2_alignedboxed,str(j), (c2x , c2y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2) #number text for each bead
					#hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
					#print(hist_mask.data)
					#hist1 = cv2.calcHist(img[c1y:c1y+7,c1x:c1x+7],[0,1,2],None,[32,32,32],[0,256])
					#print("OK")
					#print(hist1.data[4])
					#img2_aligned[c2y:c2y+7,c2x:c2x+7]

					
					newbeaddatarow = (i, mean1[2], mean1[1], mean1[0], j, mean2[2], mean2[1], mean2[0], deltaR, deltaG, deltaB)
					csvdata = csvdata + (newbeaddatarow,)

#print csvdata

#b,g,r = cv2.split(img)
#bitwiseOr = cv2.bitwise_or(b, maskadditive)
#cv2.imshow('mask1',bitwiseOr)

#print matchingBead

#### end of overlap check ######


#cv2.imshow('combinedcrops', combinedcrop1)
resizedfgcopy = cv2.resize(sure_fg,(500,400))
cv2.imshow('sure_fg bead centers',resizedfgcopy)
cv2.moveWindow('sure_fg bead centers',0,400)

totalcount = "bead count: %r"
matchcount = "match count: %r" 


#combinedimg= np.concatenate((img, img2alignedprestine), axis=1)
combinedimg= np.concatenate((img, img2_alignedboxed), axis=1)
resizedimg = cv2.resize(combinedimg,(1000,400))

height, width, channels = resizedimg.shape
cv2.putText(resizedimg,totalcount % beadcountbefore, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
cv2.putText(resizedimg,totalcount % beadcountafter, (width/2+5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
cv2.putText(resizedimg,matchcount % matchingBeadCount, (width/2+5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
cv2.imshow('Image match overlap',resizedimg)
cv2.moveWindow('Image match overlap',0,0)


#make csv file here
with open('beaddata.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(csvdata)

####

cv2.waitKey(0)
cv2.destroyAllWindows()




