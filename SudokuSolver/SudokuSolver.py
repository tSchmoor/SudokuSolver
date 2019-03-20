import sys
import numpy as np
import cv2 as cv
from Grid import Grid
from getDigitImages import getDigitalDigits

# =============================================================================
# OBJECT SudokuSolver
# =============================================================================
class SudokuSolver:
    """
    Constructor of the SudokuSolver, this method loads an image of a sudoku grid and solves it
        INPUTS:     model : the model needed to predict the digits of the cells
                    filename : the path of the fils containing the image to process
                    save(=False) : either to save the result as an image or not
                    savefile(=None) : the path of the file in which to save the resulting image, used only if save is True
                    digitImagesPath(=None) : the path of the directory in which are stored the digit images to draw on the resulting image, used only if save is True
    """
    def __init__(self, model, filename, save=False, savefile=None, digitImagesPath=None):
        self.SudokuGrid = None              # Grid used to analyse the image and solve the sudoku
        self.OutputImage = None             # Image of the solution of the sudoku
        self.OutputUnrotatedImage = None    # Unrotated image of the solution of the sudoku
        
        # Load the image
        try:
            SudokuImage = cv.imread(filename,0)
        except Exception as e:
            print("\nError during the loading of the image:")
            print(e)
            sys.exit()
        
        # Add a small margin
        try:
            (h,w) = SudokuImage.shape
            hr,wr = int(h*1.1), int(w*1.1)
            InputImage = SudokuImage.max()*np.ones((hr,wr))
            InputImage[int(np.floor((hr-h)/2)):int(np.floor((hr+h)/2)),int(np.floor((wr-w)/2)):int(np.floor((wr+w)/2))] = SudokuImage
            InputImage = np.uint8(InputImage)
        except Exception as e:
            print("\nError during the creation of the margin:")
            print(e)
            sys.exit()
        
        # Instanciate the Grid
        try:
            self.SudokuGrid = Grid(InputImage)
        except Exception as e:
            print("\nError during the instanciation of the Grid:")
            print(e)
            sys.exit()
        
        # Analyse the image
        try:
            self.SudokuGrid.setCellImages()
        except Exception as e:
            print("\nError during the analysis of the grid:")
            print(e)
            sys.exit()
        
        # Predict the digits
        try:
            self.SudokuGrid.predictDigits(model)
        except Exception as e:
            print("\nError during the prediction of the digits:")
            print(e)
            sys.exit()
        
        # Solve the Sudoku
        try:
            self.SudokuGrid.solve()
        except Exception as e:
            print("\nError during the solving of the sudoku:")
            print(e)
            sys.exit()
        
        # Save the result as an image
        try:
            if save:
                if savefile != None:
                    self.OutputImage, self.OutputUnrotatedImage = self.SudokuGrid.toImage(getDigitalDigits(digitImagesPath), savefile)
                else:
                    self.OutputImage, self.OutputUnrotatedImage = self.SudokuGrid.toImage(getDigitalDigits(digitImagesPath), filename)
            else:
                self.OutputImage, self.OutputUnrotatedImage = self.SudokuGrid.toImage(getDigitalDigits(digitImagesPath))
        except Exception as e:
            print("\nError during the saving of the solution:")
            print(e)
            sys.exit()
