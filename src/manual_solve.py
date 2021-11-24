#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

"""
Name: Lathishbabu Ganesan
ID: 21249312
GitHub Link: https://github.com/lathy88/ARC

Summary: 
1. The first two functions uses numpy N- Dimensional iterator to iterate the 2D array. This iterates each row in 
the grid and corresponding columns in each row.
2. The second program uses recursion to backtrack the rows & columns until non-zero valued cell is found.
3. The third program uses infinite loop to generate the squared boxes until the rows & columns exceed the grid boundary.
In short all 3 programs tries to solve the logic in different ways. Program 1 & 2 uses similar loop to iterate the
elements.

"""


def solve_868de0fa(x):
    """
    Required Transformation: The input can contain any number of square boxes. The program should fill the square boxes
    with Red(Colour Code = 2) or Orange(Colour Code = 7) based on the Odd or Even number of blocks(or cell) inside the
    square box respectively.

    Implementation: The solution first tries to find the boxes in the grid. Once it identifies the non-zero cell and the
    adjacent cell is 0, We need to ensure that the cell is actually the start of the block to fill. We find that out by
    checking the corner cells for non-zero values.It indicates the start of the cell where we need to fill the row
    (within the square box boundary) with respective colour.
    The second step is to figure out which colour to fill. Either it can be 2 or 7. To find that we need to calculate
    the number of cells (Cells that need to fill with colour) inside the square box. If the count is even, we will it
    with 2 or 7 otherwise.
    The logic tries to fill on row basis i.e the counter is started once it identifies the start of the cell to fill.
    After the counter value is calculated, the entire row within the square box boundary is filled with corresponding
    colour based on the counter value.(i.e Even or Odd)

    Training & Test Grid: The solution works on all Training & Test cases

    """
    for i_row, i_col in np.ndindex(x.shape):
        # if the cell value is 1 and the right adjacent cell is 0 and corner cells is non-zero then it indicates the
        # start of the square.
        if x[i_row, i_col] == 1 and (i_col + 1 < x[0].shape[0]) and (x[i_row, i_col + 1] == 0) and (
                i_row - 1 > -1) and (x[i_row - 1, i_col] != 0) and x[i_row - 1, i_col + 1] != 0:
            row, col = i_row, i_col + 1
            counter = 0
            # Calculate the number of cells inside the square box
            while col < x[0].shape[0] - 1 and x[row, col] != 1:
                col = col + 1
                counter = counter + 1
            # Fill the entire row within the square boundary with 2 or 7 based on the even/odd cells inside the square
            if counter % 2 == 0:
                col = i_col + 1
                # Even square so fill it with 2
                while x[row, col] != 1:
                    x[row, col] = 2
                    col = col + 1
            else:
                col = i_col + 1
                # Odd square so fill it with 7
                while x[row, col] != 1:
                    x[row, col] = 7
                    col = col + 1
    return x


def solve_c3f564a4(x):
    """
    Required Transformation: The input contains a specific colour pattern with some black region (Groups of cells with
    black colour). The program should identify the pattern in the grid and fill the corresponding black coloured cells
    with the suitable colour. The Grid can contain 1 or many group of black coloured region.

    Implementation: The solution is to figure out the pattern from the grid. First the program should identify the cell
    with 0 value. i.e it corresponds to cell with black colour. Once the first black coloured cell is identified, the
    program calls the support function find_right_colour to find the colour sequence of the black coloured region.
    The find_right_colour is a recursive function where it recursively go back to the previous row and next column until
    non-zero cell is encountered. This non-zero value is the colour we need to fill for the cell when the recursive
    function returns.

    Training & Test Grid: The solution works on all Training & Test cases

    """
    for i_row, i_col in np.ndindex(x.shape):
        if x[i_row, i_col] == 0:
            find_right_colour(x, i_row, i_col)
    return x


def find_right_colour(x, row, col):
    # Base case: if the cell value is non-zero, return that value
    if x[row, col] != 0:
        return x[row, col]
    if row < x.shape[0] and col > -1:
        x_row = row - 1
        y_col = col + 1
        x[row, col] = find_right_colour(x, x_row, y_col)
    return x[row, col]


def solve_5c2c9af4(x):
    """
    Required Transformation: The input contains 3 cells with non-zero value. The non-zero valued cells are diagonally
    positioned with some amount of gap between each non-zero valued cells. The program should identify the colour and
    their position in the grid and form a squared box around the centered non-zero valued cell. Each squared box should
    be of equal width between the previous one.

    Implementation: The solution is to identify the coloured cells in the grid and form a squared boxes around centered
    cell (non-zero valued cell). The Width should be same between each squared box, where width is measured by the
    difference between the number of rows or columns between 2 consecutive non-zero valued cells.
    The non-zero valued cells can be arranged in 2 forms,
    1. Up Slope
    2. Down Slope
    In the case of Up slope, once the first non-zero valued cell is identified, the pattern to fill the cells are as
    follows,
    RIGHT, DOWN, LEFT, UP
    Whereas in the case of Down Slope, once the first non-zero valued cell is identified, the pattern to fill the cells
    are as follows,
    DOWN, LEFT, UP, RIGHT
    After one full rotation, the row & column is recalculated based on the width.This process is repeated until the
    row & column goes out of the grid.

    Training & Test Grid: The solution works on all Training & Test cases

    """
    non_zero_indexes = np.nonzero(x)
    non_zero_row_array = non_zero_indexes[0]
    non_zero_col_array = non_zero_indexes[1]
    # Difference between the columns of first & second non-zero valued cell
    width = non_zero_col_array[0] - non_zero_col_array[1]
    row, col = non_zero_row_array[0], non_zero_col_array[0]
    # Centered non-zero Valued cell. This cell will become the reference point for all the squared boxes in the grid
    midpoint_loc = (non_zero_row_array[1], non_zero_col_array[1])
    value = x[non_zero_row_array[1], non_zero_col_array[1]]
    # Assign the initial width to Original Width because the width values increases as the size of the square increase.
    original_width = width
    while True:
        if width > 0:
            # Up Slope: down, left, up, right
            row, col = travel_down(x, row, col, midpoint_loc[0], abs(width), value)
            row, col = travel_left(x, row, col, midpoint_loc[1], abs(width), value)
            row, col = travel_up(x, row, col, midpoint_loc[0], abs(width), value)
            row, col = travel_right(x, row, col, midpoint_loc[1], abs(width), value)
            # Recalculate the rows & column based on the original width. Because each square should have same width
            row, col = row - abs(original_width), col + abs(original_width)
        else:
            # Down Slope: right, down, left, up
            row, col = travel_right(x, row, col, midpoint_loc[1], abs(width), value)
            row, col = travel_down(x, row, col, midpoint_loc[0], abs(width), value)
            row, col = travel_left(x, row, col, midpoint_loc[1], abs(width), value)
            row, col = travel_up(x, row, col, midpoint_loc[0], abs(width), value)
            # Recalculate the rows & column based on the original width. Because each square should have same width
            row, col = row - abs(original_width), col - abs(original_width)
        width = width + original_width
        # If the rows or columns exceed beyond the grid size terminate the loop.
        if (row < -1 and col < -1) or (row < -1 and col > x[0].shape[0]):
            break
    return x


def travel_right(x, row, col, center, width, value):
    while col < x[0].shape[0]:
        col = col + 1
        if -1 < row < x.shape[0] and col < x[0].shape[0]:
            x[row, col] = value
        # Check if the difference between column & the column of center cell is same as width, then we reached the
        # boundary for that square so break the loop
        if abs(col - center) == width:
            break
    return row, col


def travel_down(x, row, col, center, width, value):
    while row < x.shape[0]:
        row = row + 1
        if -1 < row < x.shape[0] and col < x[0].shape[0]:
            x[row, col] = value
        # Check if the difference between row & the row of center cell is same as width, then we reached the
        # boundary for that square so break the loop
        if abs(row - center) == width:
            break
    return row, col


def travel_left(x, row, col, center, width, value):
    while col > -1:
        col = col - 1
        if -1 < row < x.shape[0] and x[0].shape[0] > col > -1:
            x[row, col] = value
        # Check if the difference between column & the column of center cell is same as width, then we reached the
        # boundary for that square so break the loop
        if abs(col - center) == width:
            break
    return row, col


def travel_up(x, row, col, center, width, value):
    while row > -1:
        row = row - 1
        if row > -1 and col > -1:
            x[row, col] = value
        # Check if the difference between row & the row of center cell is same as width, then we reached the
        # boundary for that square so break the loop
        if abs(row - center) == width:
            break
    return row, col


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()
