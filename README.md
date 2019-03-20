# SudokuSolver

The goal of the SudokuSolver project was to develop a program that could solve the hardest sudoku grids in a matter of seconds starting only with the image of the grid.

This project was inspired by the "SudoCAM-Ku!" project by SanahM: https://github.com/Sanahm/SudoCAM-Ku

![alt text](https://github.com/tSchmoor/SudokuSolver/blob/master/result.png)

## How to Use

To use this program you will need a Python environment containing openCV and keras.

The SudokuSolver directory contains the main, the trained CNN model, the functions and the objects needed to analyse an image, solve a sudoku and draw the result.

In the main file, specify the name of the image containing the sudoku and the paths of the image and save directories.
You can also change the digits to use to create the output image by selecting a new directory in which to look for digit images.

The last thing you need to do is to launch the main file and your image will be processed.

## Description of the program

The SudokuSolver is based around a Grid object. A Grid object is composed of 9x9 Cell objects, 9 Line objects, 9 Column objects and 9 Square objects.

The input image is processed in order to find the grid and the image of each Cell.
Using the trained model provided, each Cell will predict its own orientation and its value. The global orientation of the grid is computed by majority voting over non-empty Cells. A new value prediction is computed for Cells which had predicted a different orientation.

The method used to solve the sudoku is based on an iterative logic inspired by human reflexion.
For each subpart of the Grid we search for a Cell that can have only one possible value or for a value that can be in only one Cell. When one of these two conditions is met, the Cell is locked to the value found.
When the iterative logic cannot further improve the solution a Backtracking algorithm is used. This is a brute force algorithm that will never fail to find a solution to a sudoku (although it is more time consuming).

When the soduku is solved the output image is created by drawing images of digits over the input image and is saved to the file specified.

## Keywords
- Image processing

- Deep learning : Convolutional Neural Network with Keras

- Computer vision
