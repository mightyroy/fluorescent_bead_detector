import pylab
import colour.plotting.diagrams as color
import csv
from colormath.color_objects import XYZColor, sRGBColor
from colormath.color_conversions import convert_color


# Plotting the *CIE 1931 Chromaticity Diagram*.
# The argument *standalone=False* is passed so that the plot doesn't get displayed
# and can be used as a basis for other plots.
color.CIE_1931_chromaticity_diagram_plot(standalone=False, grid=False)

with open('RGBoutput.csv', 'r') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     reader.next()
     for row in reader:
          R,G,B = float(row[3]), float(row[4]), float(row[5])
          rgb = sRGBColor(R, G, B)
          xyz = convert_color(rgb, XYZColor)
          X = xyz.xyz_x/ (xyz.xyz_x + xyz.xyz_y + xyz.xyz_z)
          Y = xyz.xyz_y/ (xyz.xyz_x + xyz.xyz_y + xyz.xyz_z)
          # Plotting the *xy* chromaticity coordinates.
          pylab.plot(X, Y, '.', color='black')
          

# Annotating the plot.
# pylab.annotate("PSYO",
#                xy=(0.287,0.323),
#                xytext=(-50, 30),
#                textcoords='offset points',
#                arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))


# Displaying the plot.
color.display(standalone=True)