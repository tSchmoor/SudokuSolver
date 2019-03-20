import os
## Next few lines to uncomment if there is a problem with CUDA
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import load_model
from SudokuSolver import SudokuSolver

fileName = 'Sudoku_test.png'
filePath = os.getcwd()+'\\ressources\\SudokuImages\\'+fileName
savePath = os.getcwd()+'\\ressources\\ResultsImages\\solution_'+fileName
digitImagesDirectory = os.getcwd()+'\\ressources\\digitSamples'

model_name = 'digitsModel.h5'


if __name__ == "__main__":
    model = load_model(model_name)    
    solver = SudokuSolver(model, filePath, True, savePath, digitImagesDirectory)
    print("\nSudoku Solved")