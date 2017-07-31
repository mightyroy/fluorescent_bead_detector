# fluorescent_bead_detector
OpenCV based bead detector for extracting RGB values from Oligodeoxyfluoroside library beads for Kool lab Oligodeoxyfluoroside project

Install dependencies such as OpenCV, numpy etc.

Place all bead images in same directory.

Run RoyMultiImageRGB.py in terminal from same directory as image files for all-carbon library:

![alt text](https://github.com/mightyroy/fluorescent_bead_detector/blob/master/Screen%20Shot%202017-05-30%20at%2011.20.03%20AM.png)

Output is a .csv file containing RGB values averaged from 15x15 pixel grid of center of each bead. 

For Wang Xu's two-photon library, run WangMultiImageRGBreadonly.py 

https://github.com/mightyroy/fluorescent_bead_detector/blob/master/Screen%20Shot%202017-07-05%20at%201.52.37%20PM.png
https://github.com/mightyroy/fluorescent_bead_detector/blob/master/Screen%20Shot%202017-07-05%20at%201.56.40%20PM.png

Run sRGBCIEplot.png to convert csv values into a CIE plot:

![alt text](https://github.com/mightyroy/fluorescent_bead_detector/blob/master/sRGBCIEplot.png)

Enjoy! 

Remember, manual labour is a thing of the past! 
