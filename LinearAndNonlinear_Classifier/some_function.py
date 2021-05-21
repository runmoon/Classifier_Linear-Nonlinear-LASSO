import numpy as np


class SimplifySample():

    def __init__(self, K, matrix_to_simplify):
        # remove the features that are 0 in at least K samples
        M, N = np.shape(matrix_to_simplify)
        index_not_0 = [0] * N
        for i in range(K):
            index_not_0 = index_not_0 + matrix_to_simplify[i]
        index_not_0 = np.sign(index_not_0)  # the pixel points that are 0 in at least 600 pictures are record as 0, other are 1
        self.index_not_0 = index_not_0

    def remove_0_pixel_point(self, matrix_to_simplify):
        M, N = np.shape(matrix_to_simplify)
        number_not_0 = sum(self.index_not_0)  # the number of pixel points that are not 0 in at least 600 pictures
        simplified_matrix = np.array([[0]*number_not_0] * M)
        j = 0
        for i in range(N):
            if 1 == self.index_not_0[i]:
                simplified_matrix[:, j] = matrix_to_simplify[:, i]
                j = j+1
        return simplified_matrix

# M, N = np.shape(sample_matrix)
# index_not_0 = [0] * N
# for i in range(K):
#     index_not_0 = index_not_0 + sample_matrix[i]
# index_not_0 = np.sign(index_not_0)  # the pixel points that are 0 in at least 600 pictures are record as 0, other are 1
# number_not_0 = sum(index_not_0)  # the number of pixel points that are not 0 in at least 600 pictures
# simplified_sample_matrix = np.array([[0]*number_not_0] * M)
# j = 0
# for i in range(N):
#     if index_not_0[i] == 1:
#         simplified_sample_matrix[:, j] = sample_matrix[:, i] / 1
#         j = j+1
# return simplified_sample_matrix, index_not_0

