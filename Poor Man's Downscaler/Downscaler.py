from QuickSort import *
from Distance_Ranking import *
import numpy as np


def generate_downscaled_image(low_res_image, horizontal_distance_image, vertical_distance_image, ranking_function,
                              distance_function, alpha=None):
    HR_Step = 10

    # Loading input files
    LR_FIM = low_res_image
    VD_Topo = vertical_distance_image
    HD_Topo = horizontal_distance_image

    # Initializing array for downscale results
    FIM_DS = np.zeros((64, 64))

    # Find indeces in LR_FIM where pixels are innundated
    # LR_Inundation[0] gives list row indices, LR_Inundation[1] gives list of column indices
    LR_Inundation = np.where(LR_FIM > 0)

    x_coor = LR_Inundation[0]
    y_coor = LR_Inundation[1]

    # Determine if LR_FIM contains inundated pixels
    if (LR_Inundation):
        for i in range(len(x_coor)):
            # Find HR row and cloumn corresponding to LR_FIM pixel
            row_start = ((x_coor[i] + 1) * HR_Step) - HR_Step;  # start row index for HR map
            col_start = ((y_coor[i] + 1) * HR_Step) - HR_Step;  # start column index for HR map

            # Retrieve vertical and horizontal Topography data corresponding to LR_FIM pixel
            # Fix with for loop
            HD = HD_Topo[row_start:(row_start + HR_Step),
                 col_start:(col_start + HR_Step)];  # HD data (8x8 highres), not working
            VD = VD_Topo[row_start:row_start + HR_Step,
                 col_start:col_start + HR_Step];  # VD data (8x8 highres), not working

            # Rank pixels by likelihood of flooding as determined by HD and VD
            Sort_rank = distance_ranking(HD, VD, distance_function, custom_quick_sort, alpha=alpha)

            # Apply water fraction constraint so Inundated HR pixels comply
            num_cell_flood = round(LR_FIM[x_coor[i], y_coor[i]] * pow(HR_Step, 2))

            for j in range(len(Sort_rank)):
                x_scoor = Sort_rank[j][3] + row_start
                y_scoor = Sort_rank[j][4] + col_start
                if j < num_cell_flood:
                    FIM_DS[x_scoor, y_scoor] = 1
                else:
                    FIM_DS[x_scoor, y_scoor] = 0

    return FIM_DS
