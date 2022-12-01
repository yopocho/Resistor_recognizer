# TODO: Adjust kernel sizes to suit the resolution of the 5MP RPi camera
# TODO: Add other colours of resistors
# TODO: Add silver to the band colours
# TODO: Fix DEBUG functionality for the RPi (Windows etc.)
# TODO: Make file suitable for launch at RPi startup (windows ontop, ordering, window)
# TODO: Fix resizing for DEBUG windows

"""
{Program that uses computer vision to detect the value and tolerance of a resistor
 Supports 3-5 ring values, beige 1/4W axials
 Per completetion of the course 2022_TEET-VEBEHERK-19_1_V at The University of Applied Sciences Hogeschool Utrecht
 20-10-2022}
"""
# Program info
__author__ = '{Niels van der Zijden}'
__version__ = '{1}.{1}.{0}'
__maintainer__ = '{Niels van der Zijden}'
__email__ = '{niels.vanderzijden2001@gmail.com}'
__status__ = '{Under development}'

# Libraries
from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import time

# DEBUG "define"
DEBUG = False
saveColor = "red"

# Global var that keeps track of place of exit in case of error
errorStateStep = 0

# Catch function for failed evalutions
def errorState(step):
    global errorStateStep
    errorStateStep = step
    text = "Unknown"
    cv2.putText(img=resultimg, text=text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=1)
    if DEBUG: print("Resistor of unknown value...")

# 2D list to walk over all HSV values for all colors
colors = [["black",     000, 000, 000, 180, 255,  63], 
          ["brown",      11,  97,  85,  12, 255, 135],
          ["red",       000, 104, 126,  10, 255, 255], 
          ["orange",     10, 126, 162,  15, 255, 255], 
          ["yellow",     22, 135,  67,  25, 255, 255], 
          ["green",      32,  91, 000,  45, 255, 185], 
          ["blue",       64,  60,  73,  70, 255, 147],
          ["purple",    103,  27,  44, 110, 153, 129], 
          ["grey",       10,  58, 100, 180,  79, 131],
          ["white",      19,  58, 166,  22,  77, 209], 
          ["gold",       18, 147, 000,  20, 255, 255],
          ["silver",    000, 000, 000, 000, 000, 000]] # No suitable resistor with a silver band was available during the project

# Three dictionaries to note the value of different colors in different places on the resistor
# First values
values = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "purple": 7,
    "grey": 8,
    "white": 9
    }

# Multiplier
multiplier = {
    "black": 1,
    "brown": 10,
    "red": 100,
    "orange": 1000,
    "yellow": 10000,
    "green": 100000,
    "blue": 1000000,
    "purple": 10000000,
    "grey": 100000000,
    "white": 1000000000,
    "gold": 0.1,
    "silver": 0.01
    }

# Tolerance
tolerance = {
    "brown": 1,
    "red": 2,
    "orange": 0.05,
    "yellow": 0.02,
    "green": 0.5,
    "blue": 0.25,
    "purple": 0.1,
    "grey": 0.01,
    "gold": 5,
    "silver": 10
    }

# Define E12-series; the base values of resistors that can be recognized
e12 = [10,12,15,18,22,27,33,39,47,56,68,82] # E-12 series

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
# ~ picam2.start_preview(Preview.QTGL, x=100, y=200, width = 1920, height=1080, transform=Transform(hflip=1))
picam2.start()

# Window setup
cv2.startWindowThread()
cv2.namedWindow("image")

# Main loop
while True:
    
    # Preprocessing the selected image
    img = picam2.capture_array()
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    filtered = cv2.GaussianBlur(hsv.copy(),(101,101),0)
    
    # Detect the body of the resistor and clean up the mask
    lowerColor = np.array([0,91,0])
    upperColor = np.array([180,255,255])
    out = cv2.inRange(filtered.copy(),lowerColor,upperColor)
    kernel = np.ones((101,101), np.uint8)
    dilated = cv2.dilate(out.copy(), kernel)
    eroded = cv2.erode(dilated.copy(), kernel)
    mask = eroded.copy()
    
    # Find the contour and bounding box of the mask of the resistor
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    flag = 0
    if len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect) #The most bottom point of the rectangle is the firt xy-coord, then moving clockwise
        box = np.int0(box)
        
        
        # Calculate BB-ratio
        side1 = ((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)**0.5
        side2 = ((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)**0.5
        ratio = min((side1,side2)) / max((side1, side2))
        if DEBUG:
            print("---BB aspect ratio---")
            print("side1's length = %f \nside2's length = %f" % (side1, side2))
            print("The ratio of the bounding box is %f" % ratio)
            
        # Check if the ratio of the bounding box is similar to that of a resistor
        resultimg = img.copy()
        if ratio > 0.3 and ratio < 0.42:
            if DEBUG: print("Resistor found!")
            
            # Draw the bounding box
            rect2 = cv2.drawContours(resultimg,[box],0,(0,0,255),2) 
            flag = True
            
        else:
            # Resistor not detected
            if DEBUG: print("No resistor found...")
            flag = False
        
    else:
        resultimg = np.zeros((640,480, 3), np.uint8)
    
    lineCoords = []        
    if flag: # Resistor detected
        # Cutout the detected resistor
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cutout = cv2.warpPerspective(img, M, (width, height))
        
        # Find bounding boxes of the colored lines
        hsvcutout = cv2.cvtColor(cutout.copy(), cv2.COLOR_BGR2HSV)
        filtered_hsvcutout = cv2.GaussianBlur(hsvcutout.copy(),(31,31),0)
        
        # Loop through all colors 
        j = 0
        colormask = None
        tempLineCoords = []
        for i in (range(len(colors))):
            if DEBUG: 
                print("---Current color search---")
                print("Now searching for " + colors[i][0])
                print("hsv: " + str(colors[i][1]) + "-" + str(colors[i][2]) + "-" + str(colors[i][3]) + "-" + str(colors[i][4]) + "-" + str(colors[i][5]) + "-" + str(colors[i][6]))
            
            lowerStripeColor = np.array([colors[i][1],colors[i][2],colors[i][3]])
            higherStripeColor = np.array([colors[i][4],colors[i][5],colors[i][6]])
            
            if DEBUG: 
                print("---HSV bounds---")
                print(lowerStripeColor)
                print(higherStripeColor)
            
            # Find the given color
            colormaskout = cv2.inRange(filtered_hsvcutout.copy(), lowerStripeColor, higherStripeColor)
            kernel = np.ones((61,61), np.uint8)
            dilated = cv2.dilate(colormaskout.copy(), kernel)
            eroded = cv2.erode(dilated.copy(), kernel)
            colormask = eroded.copy()
            if DEBUG:
                if colors[i][0] == saveColor:
                    specificColormask = colormask.copy()
            
            # Get the vertical lines
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            vertical = cv2.erode(colormask, verticalStructure)
            vertical = cv2.dilate(vertical, verticalStructure)                    
            
            # close holes to make it solid rectangle
            kernel = np.ones((55,55),np.uint8)
            close = cv2.morphologyEx(vertical.copy(), cv2.MORPH_CLOSE, kernel)
            
            # get contours
            contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # save the found contours
            if len(contours) > 0:
                for cnt in contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    tempLineCoords.append((colors[i][0], int(x+w/2), int(y+h/2), area))
                    if DEBUG: 
                        print("---BB area---")
                        print(area)
                
            
        # Pick the largest found areas
        tempLineCoords.sort(key=lambda tup: tup[3])
        tempLineCoords.reverse()
        
        if DEBUG:
            print("---tempLineCoords---") 
            print(tempLineCoords)
            areaValues = []
            for y in range(len(tempLineCoords)):
                areaValues.append(tempLineCoords[y][3])
            areaValues=np.array(areaValues)
        
        if len(tempLineCoords) >= 4:
            for z in range(4): # Sadly hard-coded for now
                lineCoords.append(tempLineCoords[z])
            
            # Calculate the std. dev. of the X and Y axes to find the axis along which the lines were found sorted from smallest to highest x/y coords
            lineCoordsX = []
            lineCoordsY = []
            for k in range(len(lineCoords)):
                lineCoordsX.append(lineCoords[k][1])
                lineCoordsY.append(lineCoords[k][2])
            lineCoordsXstdDev = np.std(lineCoordsX)
            lineCoordsYstdDev = np.std(lineCoordsY)
            axisIndex = None
            if lineCoordsXstdDev > lineCoordsYstdDev:
                axisIndex = 1 # lines found along X-axis
            else:
                axisIndex = 2 # lines found along Y-axis
            
            # Sort the found lines
            lineCoords.sort(key=lambda tup: tup[axisIndex])
            
            # Easy early check to find out if the colors already go left to right or the otherway
            if lineCoords[0][0] == 'gold' or lineCoords[0][0] == 'silver':
                lineCoords.reverse()
            
            # Calculate the value of the resistor
            valueString = ""
            for l in range(len(lineCoords)-2):
                valueString = valueString + str(values.get(lineCoords[l][0]))                
            resistorValue = int(valueString)
            
            # Check if the stripes were processed in the right way, or if the value is even in the e12 series
            errorFlag = False
            if resistorValue not in e12:                   
                errorFlag = True
                
            # Finish the calculation of the value and tolerance and print it out!
            if not errorFlag:
                resistorValue *= multiplier.get(lineCoords[len(lineCoords)-2][0])
                toleranceValue = tolerance.get(lineCoords[len(lineCoords)-1][0])
                text = str(resistorValue) + "ohm " + str(toleranceValue) + "%"
                cv2.putText(img=resultimg, text=text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=1)
                if DEBUG: 
                    print("---Resistor info---")
                    print("The resistor has a value of " + str(resistorValue) + "Ω and a tolerance of " + str(toleranceValue) + "%")
                
            else: # No resistor detected
                errorState(1)
                
        else: # No resistor detected
            errorState(2)
            
    else: # No resistor detected
        errorState(3)
    
    # Calculate image scale ratio
    scale_percent = 15 # percent of original size
    width = int(resultimg.shape[1] * scale_percent / 100)
    height = int(resultimg.shape[0] * scale_percent / 100) 
    ratiodim = (width, height)
    
    # Display the images
    setdim = (480, 360) # Standard 480p window size so it's easier to view images of different sizes (Aspect-ratio of the different photos is ugly I am aware)
    if DEBUG:
        
        shsv = cv2.resize(hsv, setdim, interpolation = cv2.INTER_AREA)
        cv2.imshow("HSV", shsv)
        # ~ cv2.setWindowProperty("HSV", cv2.WND_PROP_TOPMOST, 1)
        smask = cv2.resize(mask, setdim, interpolation = cv2.INTER_AREA)
        cv2.imshow("mask", smask)
        # ~ cv2.setWindowProperty("mask", cv2.WND_PROP_TOPMOST, 1)
        
        if errorStateStep > 0:
            scutout = cv2.resize(cutout, setdim, interpolation = cv2.INTER_AREA)
            cv2.imshow("cutout", scutout)
            # ~ cv2.setWindowProperty("cutout", cv2.WND_PROP_TOPMOST, 1)
            
        if errorStateStep > 1:
            scolormask = cv2.resize(specificColormask, setdim, interpolation = cv2.INTER_AREA)
            cv2.imshow(saveColor + " color mask", scolormask)
            # ~ cv2.setWindowProperty(saveColor + " color mask", cv2.WND_PROP_TOPMOST, 1)
            
    # ~ simg = cv2.resize(resultimg, ratiodim, interpolation = cv2.INTER_AREA)
    cv2.imshow("image", resultimg)
    # ~ cv2.setWindowProperty("image", cv2.WND_PROP_TOPMOST, 1)
    
    # Poll for Esc-key to exit
    key = cv2.waitKeyEx(50)
    if key != 27: # Esc
        continue
    else: # Exit program
        if DEBUG: print("\n\n\n\n\n\nGoodbye!\n")
        cv2.destroyAllWindows()
        exit(1) 
        
# End of program
