from abc import ABC, abstractmethod
from Cell import CellState

import numpy as np

# =============================================================================
# ABSTRACT OBJECT GridPart
# =============================================================================
class GridPart(ABC):
    """
    Constructor:
        INPUTS :    cellArray : array of the Cell objects in the GridPart      
                    index : the index of the GridPart
    """
    def __init__(self, cellArray, index):
        self._index = index                                             # index of the GridePart
        self._digitsFound = [False for _ in range(9)]                   # boolean array of digits founds
        self._cells = cellArray                                         # array of the Cells in the GripPart

    # =============================================================================
    # Set and Get methods used to interact with the fields of the Line object
    # =============================================================================
    """
    Set method that refresh the boolean array of digits found of the Line
    """
    def refreshDigitsFound(self):
        self._digitsFound = [False for _ in range(9)]
        for cell_index in range(9):
            value =self._cells[cell_index]._value
            if value != 0:
                self._digitsFound[value-1] = True
    """
    Set method that refresh the boolean array of potential values of every Cell of the Line
    """
    def refreshPotentials(self):
        for cell in self._cells:
            cell.refreshPotentials()
    """
    Get method that returns if the digit specified is already found in the Line
        INPUT :     digit : digit to test
        OUTPUT :    bool : True if the digit is found in the Line
    """
    def isFound(self, digit):
        return self._digitsFound[digit-1]
    
    """
    [ASTRACT] Get method that returns the values of the GridPart as a string
    """
    @abstractmethod
    def toString(self):
        pass

    # =============================================================================
    # Method used to solve the grid using the "One-possibility resolution" method (logical solution)
    # =============================================================================
    """
    Method searching for Cells that have only one potential digit or digits that are potentially in only one Cell
        OUTPUT :    improvement : True if the method managed to get an improvement on the reslution of the Grid
    """
    def solve(self):
        improvement = False
        
        for cell in self._cells :
            if cell._state == CellState.UNLOCKED:
                potentialDigits = cell.getPotentialDigits()
                if len(potentialDigits) == 1 :
                    if cell.lockTo(potentialDigits[0]):
                        improvement = True
        
        for digit in range(1,10):
            if not self.isFound(digit):
                potentialCell = []
                for cell in self._cells:
                    if cell._state == CellState.UNLOCKED and cell.isPotential(digit):
                        potentialCell.append(cell)
                if len(potentialCell) == 1:
                    if potentialCell[0].lockTo(digit):
                        improvement = True
        return improvement


# =============================================================================
# OBJECT Line
# =============================================================================
class Line(GridPart):
    """
    Constructor:
        INPUTS :    cells : the Cell array of the Grid
                    index : index of the Line in the Grid
    """
    def __init__(self, cells, index):
        cell_array = np.array([cells[index,c] for c in range(9)])       # array of the Cells in the Line
        super().__init__(cell_array, index)
        for cell in self._cells:
            cell.setLine(self)                                          # indicate to the cells to which Line they belong

    """
    Get method that returns the values of the Line as a string
        OUTPUT :    string : the string version of the Line
    """
    def toString(self):
        string = ''
        for c in range(9):
            if c%3==0:
                string+='|'
            if self._cells[c]._value !=0:
                string += '|'+str(self._cells[c]._value)
            else:
                    string += '|_'
        string += '|'
        return string



# =============================================================================
# OBJECT Column
# =============================================================================
class Column(GridPart):
    """
    Constructor:
        INPUTS :    cells : the Cell array of the Grid
                    index : index of the Column in the Grid
    """
    def __init__(self, cells, index):
        cell_array = np.array([cells[l,index] for l in range(9)])       # array of the Cells in the Column
        super().__init__(cell_array, index)
        for cell in self._cells:
            cell.setColumn(self)                                        # indicate to the cells to which column they belong

    """
    Get method that returns the values of the Column as a string
        OUTPUT :    string : the string version of the Column
    """
    def toString(self):
        string = ''
        for l in range(9):
            if l%3==0:
                string += '---\n'
            if self._cells[l]._value !=0:
                string += '|'+str(self._cells[l]._value)+'|\n'
            else:
                string += '|_|\n'
        string += '---'
        return string

    
# =============================================================================
# OBJECT Square 
# =============================================================================
class Square(GridPart):
    """
    Constructor:
        INPUTS :    cells : the Cell array of the Grid
                    index : index of the Square in the Grid
    """
    def __init__(self, cells, index):
        cell_array = np.array([cells[3 *(index//3)+k//3, 3*(index%3)+k%3] for k in range(9)])       # array of the Cells in the Square
        super().__init__(cell_array, index)
        for cell in self._cells:
            cell.setSquare(self)                                                                    # indicate to the cells to which square they belong
    
    """
    Get method that returns the values of the Square as a string
        OUTPUT :    string : the string version of the Column
    """
    def toString(self):
        string = ''
        for l in range(3):
            if l%3==0:
                string += '---------\n'
            for c in range(3):
                if c%3==0:
                    string+='|'
                if self._cells[3*l+c]._value !=0:
                    string += '|'+str(self._cells[3*l+c]._value)
                else:
                    string += '|_'
            string += '||\n'
        string += '---------'
        return string
