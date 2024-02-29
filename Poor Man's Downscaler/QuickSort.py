from numpy.core.fromnumeric import size
import numpy as np
from math import sqrt

def custom_quick_sort(horizontal_distance, vertical_distance, reference, distance_function, debug=False, alpha=None):
    # 8x8 arrays of the topography distances as passed through by Ranking Function
    HD = horizontal_distance;
    VD = vertical_distance;
    ref = reference;

    # indices of pivot point in HD and VD
    pivot_posx = 0;
    pivot_posy = 0;

    length = len(horizontal_distance)

    # Distance of pivot point from reference point
    if distance_function in "L2":
        # euclidean distance
        pivot_dist = sqrt(pow(HD[pivot_posx, pivot_posy] - ref[0], 2) + pow(VD[pivot_posx, pivot_posy] - ref[1], 2));
    elif distance_function in "L1":
        # L1 distance
        pivot_dist = (np.abs(HD[pivot_posx, pivot_posy] - ref[0]) + np.abs(VD[pivot_posx, pivot_posy] - ref[1]));
    elif distance_function in "L1-weighted":
        assert alpha is not None, "A valid input for alpha must be passed"
        # weighted L1 distance
        pivot_dist = (
                    np.abs(HD[pivot_posx, pivot_posy] - ref[0]) + alpha * np.abs(VD[pivot_posx, pivot_posy] - ref[1]));
    elif distance_function in "L2-weighted":
        assert alpha is not None, "A valid input for alpha must be passed"
        # weighted L1 distance
        pivot_dist = sqrt(
            pow(HD[pivot_posx, pivot_posy] - ref[0], 2) + alpha * pow(VD[pivot_posx, pivot_posy] - ref[1], 2));
    else:
        print("Error: improper distance method entered")

    # array for pivot containing distance from reference, HD data, VD data, and indices in HD/VD arrays
    pivot = [pivot_dist, HD[pivot_posx, pivot_posy], VD[pivot_posx, pivot_posy], pivot_posx, pivot_posy];

    # Inititialize lesser and greater arrays regarding pivot
    items_greater = [];
    items_lower = [];

    # initialize start point for count at 1 to avoid pivot at (0,0)
    # Start index will be at (0,1)
    for i in range(1, np.size(HD)):

        # The above switch statement can be replaced with the below:
        #  Note that // produces the quotient of the division
        x_coor = i // 8
        y_coor = i - (i // 8) * (8)

        # Gives vertical distance data at the indice
        VD_val = VD[x_coor, y_coor];
        HD_val = HD[x_coor, y_coor];

        # retrieves row and column distance data from reference array. This is defined in the euclidian distance function, which is passed through as parameter reference
        ref_x = ref[0];
        ref_y = ref[1];

        # returns distance of given pixel from reference point : l2 distance, euclidean distance, frobenius norm
        if distance_function == "L2":
            ref_dist = sqrt(pow(HD_val - ref_x, 2) + pow(VD_val - ref_y, 2));
        elif distance_function == "L1":
            ref_dist = (np.abs(HD_val - ref_x) + np.abs(VD_val - ref_y));
        elif distance_function == "L2-weighted":
            ref_dist = sqrt(pow(HD_val - ref_x, 2) + alpha * pow(VD_val - ref_y, 2));
        elif distance_function == "L1-weighted":
            ref_dist = (np.abs(HD_val - ref_x) + alpha * np.abs(VD_val - ref_y));
        else:
            print("Error: improper distance method entered")

        # The following statements classify a given pixel as further or closer to the reference point than the pivot
        # If closer to the reference point than pivot, the pixel reference distance, HD data, VD data, and indices are saved in items_lower
        # If further from the reference point than pivot, the pixel reference distance, HD data, VD data, and indices are saved in items_greater
        if ref_dist > pivot[0]:
            items_greater.append((ref_dist, HD_val, VD_val, x_coor, y_coor));
        elif ref_dist == pivot[0]:
            if VD_val > pivot[2]:
                items_greater.append((ref_dist, HD_val, VD_val, x_coor, y_coor));
            elif VD_val < pivot[2]:
                items_lower.append((ref_dist, HD_val, VD_val, x_coor, y_coor));
            else:
                if HD_val < pivot[1]:
                    items_lower.append((ref_dist, HD_val, VD_val, x_coor, y_coor));
                elif HD_val > pivot[1]:
                    items_greater.append((ref_dist, HD_val, VD_val, x_coor, y_coor));
        elif ref_dist < pivot[0]:
            items_lower.append((ref_dist, HD_val, VD_val, x_coor, y_coor));

        # This occurs if for some reason the pixel cannot be sorted as greater or lesser than the pivot, ie. is the pivot itself
        else:
            print("distance error, pivot: ")
            print(pivot)
            print(" HD: ")
            print(HD[x_coor, y_coor])
            print(" VD:")
            print(VD[x_coor, y_coor])

    # To return a single array, an innner sort is continued with the items_lower and items_greater arrays
    return inner_quick_sort(items_lower) + [pivot] + inner_quick_sort(items_greater);


def inner_quick_sort(item_array, debug=False):
    # If the length of the item_array is one or less to be sorted, it is considered sorted
    if len(item_array) <= 1:
        return item_array
    # If not, item_array is not sorted. The last item is set as the pivot and quicksort proceeds
    else:
        pivot = item_array[len(item_array) - 1];

    # Inititialize lesser and greater arrays regarding pivot
    it_greater = [];
    it_lower = [];

    for i in range(len(item_array) - 1):

        # The following statements classify a given pixel as further or closer to the reference point than the pivot
        # If closer to the reference point than pivot, the item is saved in items_lower
        # If further from the reference point than pivot, the item is saved in items_greater
        if item_array[i][0] > pivot[0]:
            it_greater.append(item_array[i])
        elif item_array[i][0] < pivot[0]:
            it_lower.append(item_array[i]);
        elif item_array[i][0] == pivot[0]:
            if item_array[i][2] > pivot[2]:
                it_greater.append(item_array[i]);
            elif item_array[i][2] < pivot[2]:
                it_lower.append(item_array[i]);
            else:
                if item_array[i][1] < pivot[1]:
                    it_lower.append(item_array[i]);
                elif item_array[i][2] > pivot[1]:
                    it_greater.append(item_array[i]);
        else:

            # This occurs if for some reason the pixel cannot be sorted as greater or lesser than the pivot, ie. is the pivot itself
            if debug:
                print("distance error, pivot: ")
                print(pivot)
                print("item array: ")
                print(item_array[i])

    # This can probably be deleted. Had an infinite loop resulting from sorting through to including the pivot at the end.
    if len(it_lower + [pivot]) == len(item_array):
        return ([pivot] + inner_quick_sort(it_lower))
    else:
        return (inner_quick_sort(it_lower) + [pivot] + inner_quick_sort(it_greater));