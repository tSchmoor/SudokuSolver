import numpy as np
import cv2 as cv
from Cell import Cell
from GridPart import Line, Column, Square


# =============================================================================
# OBJECT Grid
# =============================================================================
class Grid:
    """
    Constructor:
        INPUTS :    InputImage : the input image of the sudoku grid
    """
    def __init__(self, InputImage):
        self._inputImage = InputImage                                                   # Input image
        self._unrotatedImage = None                                                     # Unrotated image
        self._outputImage = None                                                        # Output Image with the same orientation as the Unrotated image
        self._unrotatedOutputImage = None                                               # Output Image with a screen orientation
        self._cells = np.array([[Cell(i,j,self) for j in range(9)] for i in range(9)])  # array of the Cells in the grid
        self._lines = np.array([Line(self._cells, k) for k in range(9)])                # array of the Lines in the grid
        self._columns = np.array([Column(self._cells, k) for k in range(9)])            # array of the Columns in the grid
        self._squares = np.array([Square(self._cells, k) for k in range(9)])            # array of the Squares in the grid
        self._orientation = None                                                        # orientation of the Grid
    
    # =============================================================================
    # Set and Get methods used to interact with the fields of the Grid object
    # =============================================================================
    """
    Set method that refresh the boolean array of digits found of every Line, Column and Square of the Grid
    """
    def refreshDigitsFound(self):
        for line in self._lines:
            line.refreshDigitsFound()
        for column in self._columns:
            column.refreshDigitsFound()
        for square in self._squares:
            square.refreshDigitsFound()
            
    """
    Set method that refresh the boolean array of potential values of every Cell of the Grid
    """
    def refreshPotentials(self):
        for line in self._lines:
            line.refreshPotentials()
    
    """
    Get method that returns the values of the Grid as a string
        OUTPUT :    string : the string version of the Grid
    """
    def toString(self):
        string = ''
        for l in range(9):
            if l%3==0:
                string += '----------------------\n'
            for c in range(9):
                if c%3==0:
                    string+='|'
                if self._cells[l,c]._value !=0:
                    string += '|'+str(self._cells[l,c]._value)
                else:
                        string += '|_'
            string += '||\n'
        string += '----------------------'
        return string
    
    """
    Get method that returns the Grid as an image
        INPUTS :    digitsImages : the digit images to use
                    filename(=None) : file to save the unrotated output image
        OUTPUTS :   self._outputImage : the output image of the Grid
                    self._unrotatedOutputImage : the output image in the orientation of the screen
    """
    def toImage(self, digitsImages, filename=None):
        # Get the unrotated image as a RGB image
        imageRGB = cv.cvtColor(255-self._unrotatedImage,cv.COLOR_GRAY2RGB)
        # Draw the values of the cells found
        for line in self._lines:
            for cell in line._cells:
                cellImage, BBox = cell.toImage(digitsImages)
                imageRGB[BBox[0]:BBox[2],BBox[1]:BBox[3],:] = cellImage
        self._outputImage = imageRGB
        # Unrotate the output image
        self._unrotatedOutputImage = self._outputImage
        if self._orientation != 0:
            self._unrotatedOutputImage = cv.rotate(self._outputImage, self._orientation-1)
        
        if filename != None :
            cv.imwrite( filename, cv.cvtColor(self._unrotatedOutputImage,cv.COLOR_RGB2BGR));
        
        return self._outputImage, self._unrotatedOutputImage
        
    """
    Set method that unrotates the input image (modulo 90°), cuts it into cells and set the Cells' images accordingly
    """
    def setCellImages(self):
        # Get negative binary image
        inputImageErode = cv.erode(self._inputImage,
                                   cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
        
        negativeImage = 255-self._inputImage
        _,negativeBinaryImage = cv.threshold(inputImageErode,
                                       215,
                                       255,
                                       cv.THRESH_BINARY_INV)
        (hn,wn) = negativeBinaryImage.shape
        
        # UNDO THE ROTATION OF THE IMAGE
        # Find the grid bounding rectangle and it's angle
        _, contours,_ = cv.findContours(negativeBinaryImage,
                                        cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
        smallestBoundingRectangle = cv.minAreaRect(contours[0])
        angle = -smallestBoundingRectangle[2]
    
        # Undo the rotation
        rotation = cv.getRotationMatrix2D((int(hn/2),int(wn/2)),
                                     angle,
                                     1)
        rotationInvertion = cv.invertAffineTransform(rotation)
        
        unrotatedBinaryImage = np.uint8(cv.warpAffine(src=negativeBinaryImage,
                                                      M=rotationInvertion,
                                                      dsize=(hn,wn),
                                                      borderMode=cv.BORDER_CONSTANT,
                                                      borderValue=0))
        self._unrotatedImage = np.uint8(cv.warpAffine(src=negativeImage,
                                                M=rotationInvertion,
                                                dsize=(hn,wn),
                                                borderMode=cv.BORDER_CONSTANT,
                                                borderValue=0))
        
        # FIND THE CELLS IN THE IMAGE
        # Copy the grid using HoughLines
        houghLinesP = cv.HoughLinesP(unrotatedBinaryImage,
                                     rho=1,
                                     theta=np.pi/16,
                                     threshold=np.uint16((hn+wn)/8),
                                     minLineLength=np.uint16((hn+wn)/20),
                                     maxLineGap=np.uint16((hn+wn)/20))
        
        gridImage = np.zeros(unrotatedBinaryImage.shape, dtype = np.uint8)
        for idx in range(houghLinesP.shape[0]):
            cv.line(gridImage,
                     (houghLinesP[idx,0,0],houghLinesP[idx,0,1]),
                     (houghLinesP[idx,0,2],houghLinesP[idx,0,3]),
                     color=255)

        # Get connected components of the copied grid
        NegativeGrid = 255-gridImage
        numberLabels,_,componentsStats,_ = cv.connectedComponentsWithStats(NegativeGrid)
        
        # Set the area threshold to remove the background and the grid, i.e. to keep only the cells
        areaMax = hn*wn/81
        areaMin = hn*wn/(810)
        
        # Get the lines and column intervals
        lignes = []
        columns = []
        count = 0
        for i in range(numberLabels):
            if componentsStats[i][4]<areaMax and componentsStats[i][4]>areaMin:
                if count//9 == 0:
                    columns.append((componentsStats[i][0],componentsStats[i][0]+componentsStats[i][2]))
                if count%9 == 0:
                    lignes.append((componentsStats[i][1],componentsStats[i][1]+componentsStats[i][3]))
                count +=1 
                
        # SET THE CELLS' IMAGES
        for i in range(numberLabels):
            if componentsStats[i][4]<areaMax and componentsStats[i][4]>areaMin:
                # Get the cell bounding box and centroid
                hstart = componentsStats[i][1]
                hend = componentsStats[i][1]+componentsStats[i][3]
                centroidh = (hstart+hend)/2
                wstart = componentsStats[i][0]
                wend = componentsStats[i][0]+componentsStats[i][2]
                centroidw = (wstart+wend)/2
    
                # Get the localisation of the cell in the grid
                l,c = 0,0
                while centroidh>lignes[l][1]:
                    l+=1
                while centroidw>columns[c][1]:
                    c+=1
                # Get the image of the cell
                squareImage = self._unrotatedImage[hstart:hend,wstart:wend]
                
                # Enters the cell image in the corresponding Cell object
                self._cells[l,c].setImage(squareImage, [hstart, wstart, hend, wend])
    
    # =============================================================================
    # Predict Digits
    # =============================================================================
    """
    Method used to predict the initial values of the Grid and its orientation based on the image and the model provided to it
       INPUTS :     model : model used for the classification of digits
    """
    def predictDigits(self, model):
        # First prediction to predict the global orientation
        orientation_array = np.zeros(4)
        for line in self._lines:
            for cell in line._cells:
                orientation = cell.predictDigit(model)
                if orientation != None:
                    orientation_array[orientation]+=1
        self._orientation = np.argmax(orientation_array)
        # Second prediction on cells with value 0 or orientation different of the Grid's
        for line in self._lines:
            for cell in line._cells:
                if cell._orientation != self._orientation:
                    cell.predictDigit(model, self._orientation)
                    
    # =============================================================================
    # Method used to solve the grid using both :
    #   => the "One-possibility resolution" method (logical solution)
    #   => the "Track resolution" method (brute force)
    # =============================================================================
    """
    Method that start solving the grid using the "One-possibility resolution" method
    and end using the "Track resolution" method to find the remaining values
        OUTPUTS :   True if the grid is solved, False if it isn't
    """    
    def solve(self):
        logicalImprovement = True  
        self.refreshDigitsFound()
        self.refreshPotentials()
        # One-possibility resolution
        while(logicalImprovement):
            logicalImprovement = False
            for k in range(9):
                if self._lines[k].solve():
                    logicalImprovement = True
                if self._columns[k].solve():
                    logicalImprovement = True
                if self._squares[k].solve():
                    logicalImprovement = True
        # Track resolution
        return self._cells[0,0].track()
