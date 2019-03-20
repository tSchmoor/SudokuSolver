from enum import Enum
import numpy as np
import cv2 as cv

# =============================================================================
# ENUMERATION Cell State
# =============================================================================
class CellState(Enum):
    INITIAL = 1
    LOCKED = 2
    UNLOCKED = 3

# =============================================================================
# OBJECT Cell
# =============================================================================
class Cell:
    """
    Constructor:
        INPUTS :    l : the line of the Cell in the Grid
                    c : the column of the Cell in the Grid
                    grid : the Grid object to which the Cell belongs
    """
    def __init__(self, l, c, grid):
        self._state = CellState.UNLOCKED                        # state of the cell : INITIAL, LOCKED or UNLOCKED
        self._value = 0                                         # value of the cell 1->9
        self._potentials = [True for _ in range(9)]             # boolean array of potential values
        self._image = np.zeros((28,28),dtype=np.uint8)          # input image of the cell
        self._imageBoundingBox = None                           # Bounding box of the image : [hstart, wstart, hend, wend]
        self._output_image = np.zeros((28,28),dtype=np.uint8)   # output image of the cell
        self._orientation = None                                # counterclockwise orientation of the grid : 0:0°, 1:90°, 2:180°, 3:270°
        self._l = l                                             # line of the cell 0->8
        self._c = c                                             # column of the cell 0->8
        self._line = None                                       # Line object containing the cell
        self._column = None                                     # Column object containing the cell
        self._square = None                                     # Square object containing the cell
        self._grid = grid                                       # Grid object containing the cell

    # =============================================================================
    # Set and Get methods used to interact with the fields of the Cell object
    # =============================================================================
    """
    Set method for the _line field of the Cell
        INPUT :     line : the Line object to which the Cell belongs
    """
    def setLine(self, line):
        self._line = line
        
    """
    Set method for the _column field of the Cell
        INPUT :     column : the Column object to which the Cell belongs
    """
    def setColumn(self, column):
        self._column = column
        
    """
    Set method for the _square field of the Cell
        INPUT :     square : the Square object to which the Cell belongs
    """
    def setSquare(self, square):
        self._square = square
    
    """
    Set method for the _image field of the Cell
        INPUTS :    image : the image to set to the Cell
                    boundingBox : the bounding box of the image
    """
    def setImage(self, image, boundingBox):
        self._image = cv.resize(image,(28,28))
        self._output_image = cv.cvtColor(255-image,cv.COLOR_GRAY2RGB)
        self._imageBoundingBox = boundingBox

    """
    Get method that returns the list of potential digits of the Cell
        OUTPUT :    potentialDigits : list of potential digits of the Cell
    """
    def getPotentialDigits(self):
        potentialDigits = []
        for digit in range(1,10):
            if self._potentials[digit-1]:
                potentialDigits.append(digit)
        return potentialDigits
    
    """
    Get method that returns if the digit specified is a potential digit of the Cell
        INPUT :     digit : the digit to test
        OUTPUT :    bool : True is the digit is a potential digit of the Cell
    """
    def isPotential(self, digit):
        return self._potentials[digit-1]
    
    """
    Get method that returns the next Cell of the grid or None if the end of the Grid is reached
        OUTPUT :    Cell : the next cell of the Grid, None if the end of the Grid
    """
    def getNextCell(self):
        if self._c < 8:
            return self._grid._cells[self._l, self._c+1]    # next cell in the line
        else:
            if  self._l < 8:
                return self._grid._cells[self._l+1, 0]  # first cell of the next line
            else :
                return None     # None i.e 'end of the grid'
    
    """
    Set method that refresh the boolean array of potential values of the Cell
    """
    def refreshPotentials(self):
        if self._state == CellState.UNLOCKED:
            self._potentials = [True for _ in range(9)]
            for digit in range(9):
                if self._line.isFound(digit) or self._column.isFound(digit) or self._square.isFound(digit):
                    self._potentials[digit-1] = False
                    
    """
    Get method that returns the value of the Cell as an image given the digit images provided
        INPUT :     digitsImages : array of 9 images, one for each value
        OUTPUTS :   self._output_image : the value of the Cell as a blue image
                    self._imageBoundingBox : the bounding box of the image
    """
    def toImage(self, digitsImages):
            if self._state != CellState.INITIAL:
                digit = self._value
                digitImage = np.zeros((28,28),dtype=np.uint8)
                if digit !=0 :
                    # Select the digit image
                    digitImage = digitsImages[digit-1]
                    # If needed, apply a rotation
                    if self._orientation != 0:
                        digitImage = cv.rotate(digitImage, 3-self._orientation)
                # Compute the negative image
                digitImage = 255-digitImage
                # Compute the RGB image
                digitImageRGB = cv.cvtColor(digitImage,cv.COLOR_GRAY2RGB)
                # Compute the color modification
                if self._state == CellState.LOCKED or self._state == CellState.UNLOCKED:
                    digitImageRGB[:,:,2][digitImageRGB[:,:,2]<200] = 200
                # Resize to the input bounding box
                BBox = self._imageBoundingBox
                digitImageRGB = cv.resize(digitImageRGB,(BBox[3]-BBox[1],BBox[2]-BBox[0]))
                self._output_image = digitImageRGB
            return self._output_image, self._imageBoundingBox

    # =============================================================================
    # Predict Digits
    # =============================================================================
    """
    Method used to predict the value of a digit based on the image and the model provided to it
       INPUTS :     model : model used for the classification of digits
                    orientation(=None) : orientation of the grid, if None, test all 90° rotations
       OUTPUTS :    orientation : returns the predicted orientation of the cell, None if the value is 0 (i.e. the cell is empty)
    """
    def predictDigit(self, model, orientation=None):
        if orientation == None:     # Prediction of the orientation
            image90 = cv.rotate(self._image, cv.ROTATE_90_CLOCKWISE)
            image180 = cv.rotate(self._image, cv.ROTATE_180)
            image270 = cv.rotate(self._image, cv.ROTATE_90_COUNTERCLOCKWISE)
            
            prediction = np.array([model.predict(self._image[np.newaxis,:,:,np.newaxis]), model.predict(image90[np.newaxis,:,:,np.newaxis]), model.predict(image180[np.newaxis,:,:,np.newaxis]), model.predict(image270[np.newaxis,:,:,np.newaxis])])
            probability = np.max(prediction[:,0,0,0],axis=1)
            self._orientation = int(np.argmax(probability))
            if np.max(probability)>0.2: self._value = np.argmax(prediction[self._orientation,0,0,0])
            else: self._value = 0
        else :  # Orientation is set
            image = self._image
            if orientation!=0:
                image = cv.rotate(self._image, orientation-1)
            prediction = model.predict(image[np.newaxis,:,:,np.newaxis])
            self._orientation = orientation
            if np.max(prediction)>0.2: self._value = np.argmax(prediction)
            else : self._value = 0
        
        if self._value != 0: # Lock the Cell as INITIAL and return the orientation predicted
            self._state = CellState.INITIAL
            self._potentials = [False for _ in range(9)]
            return self._orientation
        
        self._state = CellState.UNLOCKED    # Unlock the Cell and return None
        return None
    
    # =============================================================================
    #     Methods of the "One-possibility resolution" method (logical solution)
    # =============================================================================
    """
    Method that locks the Cell to a given value then refresh the digit found and potentials of the whole grid
        INPUT :     value : value to which lock the Cell
        OUTPUT :    bool : True if the Cell was correctly locked
    """
    def lockTo(self, value):
        if self._state == CellState.UNLOCKED:
            self._state = CellState.LOCKED
            self._value = value
            self._potentials = [False for _ in range(9)]
            
            self._grid.refreshDigitsFound()
            self._grid.refreshPotentials()
            return True
        return False
        
    # =============================================================================
    #     Methods of the "Track resolution" method (brute force)
    # =============================================================================
    """
    Method that iterates the "Track resolution" method
        OUTPUT :    True if the track is finished (end of the track)
                    False if the track cannot be finished with the previous Cell's values (backtrack)
    """
    def track(self):
        if self._state == CellState.UNLOCKED:
            self._line.refreshDigitsFound()
            self._column.refreshDigitsFound()
            self._square.refreshDigitsFound()
            self.refreshPotentials()
            for digit in range(1,10):
                if self.isPotential(digit):
                    self._value = digit
                    nextCell = self.getNextCell()
                    if nextCell==None:
                        return True
                    elif nextCell.track():
                        return True
                    else:
                        self._line.refreshDigitsFound()
                        self._column.refreshDigitsFound()
                        self._square.refreshDigitsFound()
                        self.refreshPotentials()
            self._value = 0
            return False
        else :
            nextCell = self.getNextCell()
            if nextCell == None:
                return True
            else :
                return nextCell.track()
