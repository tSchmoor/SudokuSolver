import os
import cv2 as cv

"""
Loads every image in the folder specified
    INPUT:  folder : folder in which to read the images
    
    OUTPUTS:    images : the images that were loaded
                filenames : the names of the images loaded
"""
def LoadImagesFromFolder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            filenames.append(filename)
            images.append(img)
    return images, filenames
