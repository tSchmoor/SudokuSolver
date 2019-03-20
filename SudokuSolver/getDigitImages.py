import numpy as np
import cv2 as cv
from LoadImagesFromFolder import LoadImagesFromFolder

"""
Function used to get an image of a digit for each value of the sudoku
It needs at least one sample for each digit between 1 and 9
    INPUT :     digitImagesPath : the path of the directory in which are stored the digit images to load
    OUTPUT :    digitArray : array of the images
"""
def getDigitalDigits(digitImagesPath):
    # Load the images of the ressource\digitSamples directory
    DigitalImages, DigitalNames = LoadImagesFromFolder(digitImagesPath)
    DigitalImagesShape = np.shape(DigitalImages)
    
    # Find the name of each sample
    x_digital = np.zeros(DigitalImagesShape[:3],np.uint8)
    y_digital = np.zeros(DigitalImagesShape[0],np.uint8)
    for index in range(DigitalImagesShape[0]):
        image, name = DigitalImages[index], DigitalNames[index]
        Sample = 255-cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        for digit in range(1,10):
            if str(digit) in name:
                Label = digit
                x_digital[index] = np.uint8(Sample)
                y_digital[index] = np.uint8(Label)
    
    # Create the digitArray
    digitArray = np.array([None for _ in range(9)])
    for digit in range(1,10):
        numberOfSamples=0
        index = 0
        while numberOfSamples<1:
            if y_digital[index] == digit:
                digitArray[digit-1] = x_digital[index]
                numberOfSamples += 1
            index +=1
    
    return digitArray
