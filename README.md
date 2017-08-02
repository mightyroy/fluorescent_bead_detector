# Fluorescent Bead Detector

OpenCV based bead detector for extracting RGB values from Oligodeoxyfluoroside library beads (130 um, tentagel, Merck Novasyn TG) for Kool lab Oligodeoxyfluoroside project

Install dependencies such as OpenCV, numpy etc.

Place all bead images in same directory.

Run RoyMultiImageRGB.py in terminal from same directory as image files for all-carbon library. Green regions are successfully detected bead centers, red regions are overlapping bead regions ignored by the software:

![alt text](https://github.com/mightyroy/fluorescent_bead_detector/blob/master/Screen%20Shot%202017-05-30%20at%2011.20.03%20AM.png)

Output is a .csv file containing RGB values averaged from 15x15 pixel grid of center of each bead. 

For Wang Xu's two-photon library, run WangMultiImageRGBreadonly.py 

![alt text](https://github.com/mightyroy/fluorescent_bead_detector/blob/master/Screen%20Shot%202017-07-05%20at%201.52.37%20PM.png)

![alt text](https://github.com/mightyroy/fluorescent_bead_detector/blob/master/Screen%20Shot%202017-07-05%20at%201.56.40%20PM.png)

Run sRGBCIEplot.png to convert csv values into a CIE plot:

![alt text](https://github.com/mightyroy/fluorescent_bead_detector/blob/master/sRGBCIEplot.png)

For bead color change studies, put before and after images in same folder, Run pickbeadcsv.py to spatially align both images and compare bead RGB values among beads. Delta RGB values of each bead are stored in the generated csv file. Green boxes are bead centers from first image, and red boxes are bead centers from second aligned image:

![](https://github.com/mightyroy/fluorescent_bead_detector/blob/master/beadoverlap.png)


Enjoy! 

Remember, manual labour is a thing of the past! 
