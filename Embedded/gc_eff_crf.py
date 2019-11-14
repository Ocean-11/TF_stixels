
'''
*
* crf
*
* Purpose: the module finds the minimum energy semi-connected path between 1st and last row
*
* Inputs: observations_matrix
          N = energy "deadzone" for small transitions
          T = threshold to allow large transitions (between obstacles)
          W_trans =
*
* Outputs:
*
* Conventions: (x=0, y=0) is the upper left corner of the image
*
* Written by: Ran Zaslavsky 10-12-2018 (framework originates from excellent https://crosleythomas.github.io/blog/)
*
'''

# from __future__ import absolute_import

import numpy as np
import time
import constant
import copy

##############################################################
###   Transition weights matrix algorithm implementation   ###
##############################################################

# Creates a rectangular matrix weighting the transition between cell xi to cell xj
# dim = matrix width/height
# N =  "deadzone" for small enough transitions
# T =  threshold to allow transition between obstacles
# Wb = transition cost weight (in relation to Wu - the unary energy)

def create_transition_matrix(dim, N, T, Wb):

    transition_mat = np.zeros((dim ,dim))
    for row in range(dim):
        for column in range (dim):
            transition_mat[column ,row] = max ((abs(column -row)) - N, 0)
            transition_mat[column, row] = min(transition_mat[column ,row], T)
            transition_mat[column, row] = transition_mat[column ,row] * Wb

    #print(transition_mat)
    return transition_mat


######################################################
###   Efficient Viterbi algorithm implementation   ###
######################################################

# Implement an efficient Viterbi dynamic programming algorithm, assuming that only 1 cell in a row is occupied
# Go through the rows and compute the following Viterbi function: Wu*P + Wb*min(max((|y1-y2|-N,0),T)
# Inputs:
# ------
# observations_matrix - a vector holding the border cells
# transition matrix - a rectangular matrix holding the energy required to transit from 1 cell to another
# row_width - represents the stixel width
# Output:
# ------
#

def viterbi_eff_0(observations_matrix, transition_matrix, row_width):

    observations_shape = np.shape(observations_matrix)
    rows_num = observations_shape[0]
    #print('row width = {}, number of rows = {}'.format(row_width, rows_num))

    # Init params
    new_energy_vec = np.zeros((row_width, 1))
    energy_vec_a = np.ones((1,row_width)) * constant.UNOCCUPIED_CELL_ENERGY
    energy_vec_a[0,int(observations_matrix[0] )] = 0
    energy_vec = energy_vec_a[0]
    path_matrix = np.zeros((rows_num, row_width))

    start_time = time.time()

    # Go through each row
    for to_row in range(1, rows_num):

        # init the occupied cell
        max_cell_index = int(observations_matrix[to_row])
        if (max_cell_index == -1):
            # In case there is no chosen cell
            max_cell_index = int(row_width-1)
        #print('\nanalyze transitions to row {}, max cell = {}'.format(to_row, max_cell_index))

        # Go through each cell and find the lowest energy path to it
        for to_cell in range(row_width):

            # calculate transition energy vector (using lists)
            transition_energy_vec = energy_vec[:] + transition_matrix[:, to_cell]
            # find the minimum transition energy cells array
            min_energy = min(transition_energy_vec)
            min_energy_cell_num_vec = [i for i,x in enumerate(transition_energy_vec) if x==min_energy]
            # calculate the distance to occupied cell
            min_energy_dist_vec = []
            min_energy_dist_vec[:] = [abs(x - max_cell_index) for x in min_energy_cell_num_vec]
            # find the minimum distance cell
            min_energy_closest = min(min_energy_dist_vec)
            min_energy_cell_num = min_energy_cell_num_vec[min_energy_dist_vec.index(min_energy_closest)]
            path_matrix[to_row, to_cell] = int(min_energy_cell_num)

            # Update new energy vec & path
            if (to_cell == observations_matrix[to_row]):
                # Moving to the occupied border cell
                new_energy_vec[to_cell] = min_energy
            else:
                # moving to un-occupied cell
                new_energy_vec[to_cell] = min_energy + constant.UNOCCUPIED_CELL_ENERGY

        # Save new energy vec to energy vec
        for index, val in enumerate(new_energy_vec):
            energy_vec[index] = val

    #print("best path time = {}".format(time.time() - start_time))

    # find the best path
    #print('Find the best trail from end point to first one (max probability cell {}):'.format(max_cell_index))
    a = min(energy_vec)
    end_of_trail = -1
    for index, i in enumerate(energy_vec):
        if (i == a):
            if (end_of_trail == -1):
                # first min - update end cell
                end_of_trail = index
            elif (abs(index - max_cell_index) < abs(end_of_trail - max_cell_index)):
                # choose the index closest to the max-probability cell
                end_of_trail = index

    # Go through the path_matrix and create the (reverse) optimal path
    best_path_r = []
    best_path_r.append(end_of_trail)
    # print(end_of_trail)
    for row_num in range(rows_num - 1, 0, -1):
        end_of_trail = int(path_matrix[row_num, end_of_trail])
        best_path_r.append(end_of_trail)
        #print(end_of_trail)

    # Reverse the list
    best_path = list(reversed(best_path_r))
    #print(best_path)

    return best_path








def viterbi_eff_1(observations_matrix, transition_matrix, row_width):

    observations_shape = np.shape(observations_matrix)
    rows_num = observations_shape[0]
    #print('row width = {}, number of rows = {}'.format(row_width, rows_num))

    # Init params
    new_energy_vec = np.zeros((row_width, 1))
    energy_vec_a = np.ones((1,row_width)) * constant.UNOCCUPIED_CELL_ENERGY
    energy_vec_a[0,int(observations_matrix[0] )] = 0
    energy_vec = energy_vec_a[0]
    path_matrix = np.zeros((rows_num, row_width))

    start_time = time.time()

    # Go through each row
    for to_row in range(1, rows_num):

        # init the occupied cell
        max_cell_index = int(observations_matrix[to_row])
        if (max_cell_index == -1):
            # In case there is no chosen cell
            max_cell_index = int(row_width-1)
        #print('\nanalyze transitions to row {}, max cell = {}'.format(to_row, max_cell_index))

        # Go through each cell and find the lowest energy path to it
        for to_cell in range(row_width):

            # calculate transition energy vector (using lists)
            transition_energy_vec = energy_vec[:] + transition_matrix[:, to_cell]
            # find the minimum transition energy cells array
            min_energy = min(transition_energy_vec)
            min_energy_cell_num_vec = [i for i,x in enumerate(transition_energy_vec) if x==min_energy]
            # calculate the distance to occupied cell
            min_energy_dist_vec = []
            min_energy_dist_vec[:] = [abs(x - max_cell_index) for x in min_energy_cell_num_vec]
            # find the minimum distance cell
            min_energy_closest = min(min_energy_dist_vec)
            min_energy_cell_num = min_energy_cell_num_vec[min_energy_dist_vec.index(min_energy_closest)]
            path_matrix[to_row, to_cell] = int(min_energy_cell_num)

            # Update new energy vec & path
            new_energy_vec[to_cell] = min_energy + constant.UNOCCUPIED_CELL_ENERGY * (int(to_cell != observations_matrix[to_row]))

        # Save new energy vec to energy vec
        for index, val in enumerate(new_energy_vec):
            energy_vec[index] = val


        #energy_vec[:] = new_energy_vec[0][:]

    #print("best path time = {}".format(time.time() - start_time))

    # find the best path
    #print('Find the best trail from end point to first one (max probability cell {}):'.format(max_cell_index))
    a = min(energy_vec)
    end_of_trail = -1
    for index, i in enumerate(energy_vec):
        if (i == a):
            if (end_of_trail == -1):
                # first min - update end cell
                end_of_trail = index
            elif (abs(index - max_cell_index) < abs(end_of_trail - max_cell_index)):
                # choose the index closest to the max-probability cell
                end_of_trail = index

    # Go through the path_matrix and create the (reverse) optimal path
    best_path_r = []
    best_path_r.append(end_of_trail)
    # print(end_of_trail)
    for row_num in range(rows_num - 1, 0, -1):
        end_of_trail = int(path_matrix[row_num, end_of_trail])
        best_path_r.append(end_of_trail)
        #print(end_of_trail)

    # Reverse the list
    best_path = list(reversed(best_path_r))
    #print(best_path)

    return best_path

##########################################################
###   Efficient Viterbi (2) algorithm implementation   ###
##########################################################

def viterbi_eff_2(observations_matrix, transition_matrix, row_width):

    observations_shape = np.shape(observations_matrix)
    rows_num = observations_shape[0]
    #print('row width = {}, number of rows = {}'.format(row_width, rows_num))

    # Init params
    new_energy_vec = np.zeros((row_width, 1))
    energy_vec_a = np.ones((1,row_width)) * constant.UNOCCUPIED_CELL_ENERGY
    energy_vec_a[0,int(observations_matrix[0] )] = 0
    energy_vec = energy_vec_a[0]
    path_matrix = np.zeros((rows_num, row_width))

    start_time = time.time()

    # Go through each row
    for to_row in range(1, rows_num):

        # init the occupied cell
        max_cell_index = int(observations_matrix[to_row])
        if (max_cell_index == -1):
            # In case there is no chosen cell
            max_cell_index = int(row_width-1)
        #print('\nanalyze transitions to row {}, max cell = {}'.format(to_row, max_cell_index))

        # Go through each cell and find the lowest energy path to it
        for to_cell in range(row_width):

            # calculate transition energy vector (using lists)
            transition_energy_vec = energy_vec[:] + transition_matrix[:, to_cell]
            # find the minimum transition energy cells array
            min_energy = min(transition_energy_vec)
            min_energy_cell_num_vec = [i for i,x in enumerate(transition_energy_vec) if x==min_energy]
            # calculate the distance to occupied cell
            min_energy_dist_vec = []
            min_energy_dist_vec[:] = [abs(x - max_cell_index) for x in min_energy_cell_num_vec]
            # find the minimum distance cell
            min_energy_closest = min(min_energy_dist_vec)
            min_energy_cell_num = min_energy_cell_num_vec[min_energy_dist_vec.index(min_energy_closest)]
            path_matrix[to_row, to_cell] = int(min_energy_cell_num)

            # Update new energy vec & path
            new_energy_vec[to_cell] = min_energy + constant.UNOCCUPIED_CELL_ENERGY * (int(to_cell != observations_matrix[to_row]))

        # Save new energy vec to energy vec
        energy_vec = copy.copy(new_energy_vec.T[0])

    #print("best path time = {}".format(time.time() - start_time))

    # find the best path
    #print('Find the best trail from end point to first one (max probability cell {}):'.format(max_cell_index))
    a = min(energy_vec)
    end_of_trail = -1
    for index, i in enumerate(energy_vec):
        if (i == a):
            if (end_of_trail == -1):
                # first min - update end cell
                end_of_trail = index
            elif (abs(index - max_cell_index) < abs(end_of_trail - max_cell_index)):
                # choose the index closest to the max-probability cell
                end_of_trail = index

    # Go through the path_matrix and create the (reverse) optimal path
    best_path_r = []
    best_path_r.append(end_of_trail)
    # print(end_of_trail)
    for row_num in range(rows_num - 1, 0, -1):
        end_of_trail = int(path_matrix[row_num, end_of_trail])
        best_path_r.append(end_of_trail)
        #print(end_of_trail)

    # Reverse the list
    best_path = list(reversed(best_path_r))
    #print(best_path)

    return best_path


##########################################################
###   Efficient Viterbi (c) algorithm implementation   ###
##########################################################

def viterbi_eff_3(observations_matrix, transition_matrix, row_width):

    observations_shape = np.shape(observations_matrix)
    rows_num = observations_shape[0]
    #print('row width = {}, number of rows = {}'.format(row_width, rows_num))

    # Init params
    new_energy_vec = np.zeros((row_width, 1))
    energy_vec_a = np.ones((1,row_width)) * constant.UNOCCUPIED_CELL_ENERGY
    energy_vec_a[0,int(observations_matrix[0] )] = 0
    energy_vec = energy_vec_a[0]
    path_matrix = np.zeros((rows_num, row_width))

    start_time = time.time()

    # Go through each row
    for to_row in range(1, rows_num):

        # init the occupied cell
        max_cell_index = int(observations_matrix[to_row])
        if (max_cell_index == -1):
            # In case there is no chosen cell
            max_cell_index = int(row_width-1)
        #print('\nanalyze transitions to row {}, max cell = {}'.format(to_row, max_cell_index))

        # Go through each cell and find the lowest energy path to it
        for to_cell in range(row_width):

            # calculate transition energy vector (using lists)
            transition_energy_vec = energy_vec[:] + transition_matrix[:, to_cell]
            # find the minimum transition energy cells array
            min_energy = min(transition_energy_vec)
            min_energy_cell_num_vec = [i for i, x in enumerate(transition_energy_vec) if x == min_energy]
            min_energy_cell_num_1 = min(min_energy_cell_num_vec, key=lambda x: abs(x - max_cell_index))
            path_matrix[to_row, to_cell] = int(min_energy_cell_num_1)

            # Update new energy vec & path
            new_energy_vec[to_cell] = min_energy + constant.UNOCCUPIED_CELL_ENERGY * (int(to_cell != observations_matrix[to_row]))

        # Save new energy vec to energy vec
        energy_vec = copy.copy(new_energy_vec.T[0])

    #print("best path time = {}".format(time.time() - start_time))

    # find the best path
    #print('Find the best trail from end point to first one (max probability cell {}):'.format(max_cell_index))
    a = min(energy_vec)
    end_of_trail = -1
    for index, i in enumerate(energy_vec):
        if (i == a):
            if (end_of_trail == -1):
                # first min - update end cell
                end_of_trail = index
            elif (abs(index - max_cell_index) < abs(end_of_trail - max_cell_index)):
                # choose the index closest to the max-probability cell
                end_of_trail = index

    # Go through the path_matrix and create the (reverse) optimal path
    best_path_r = []
    best_path_r.append(end_of_trail)
    # print(end_of_trail)
    for row_num in range(rows_num - 1, 0, -1):
        end_of_trail = int(path_matrix[row_num, end_of_trail])
        best_path_r.append(end_of_trail)
        #print(end_of_trail)

    # Reverse the list
    best_path = list(reversed(best_path_r))
    #print(best_path)

    return best_path




#############################################################
###   Visualizing predictions and creating output video   ###
#############################################################

def main(grid, grid_width, N, T, W_trans):

    transition_matrix = create_transition_matrix(grid_width, N, T, W_trans)
    #print(np.shape(transition_matrix))

    # Use CRF (1) to find the best path
    start_time = time.time()
    for i in range(10):
        best_path = viterbi_eff_0(grid.T, transition_matrix, grid_width)
    print('0) elapsed time = {}'.format(time.time() - start_time))

    # Use CRF (1) to find the best path
    start_time = time.time()
    for i in range(10):
        best_path = viterbi_eff_1(grid.T, transition_matrix, grid_width)
    print('1) elapsed time = {}'.format(time.time() - start_time))

    # Use CRF (2) to find the best path
    start_time = time.time()
    for i in range(10):
        best_path = viterbi_eff_2(grid.T, transition_matrix, grid_width)
    print('2) elapsed time = {}'.format(time.time() - start_time))

    # Use CRF (3) to find the best path
    start_time = time.time()
    for i in range(10):
        best_path = viterbi_eff_3(grid.T, transition_matrix, grid_width)
    print('3) elapsed time = {}'.format(time.time() - start_time))

    for index, path in enumerate(best_path):
        print('{}: {}'.format(index,path))

if __name__ == '__main__':

    # define a grid example
    rows = 50
    columns = 50

    grid = -(np.ones((rows)))
    grid[0] = 0
    grid[1] = 2
    grid[2] = 3
    grid[3] = 2
    grid[4] = 3
    grid[5] = 4
    grid[40] = 40
    grid[49] = 49

    main(grid.T, columns, N=5, T=10, W_trans=5)
