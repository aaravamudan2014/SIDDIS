import numpy as np

def distance_ranking(horizontal_distance_image, vertical_distance_image, distance_function, custom_sort_function,
                     alpha=None):
    # Finding the minimum horizontal and vertical distances in the 8x8 area
    HD_min = np.amin(horizontal_distance_image);
    VD_min = np.amin(vertical_distance_image);

    # setting reference point to calculate distance from the HD and VD mins
    ref = [HD_min, VD_min];

    # Sorting each pixel by its distance from the reference point
    sorted_array = custom_sort_function(horizontal_distance_image, vertical_distance_image, ref, distance_function,
                                        alpha=alpha);

    return sorted_array;