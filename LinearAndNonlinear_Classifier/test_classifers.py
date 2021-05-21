import numpy as np
import loadfile
from math import *
import some_function

# the parameters that you can change
construct_feature = 0  # determine whether construct extra features
testSet = 1  # test=1,test set; test=0,train set;
linear = 0  # linear=1, linear least square; linear=0, nonlinear least square

imgs1, head1 = loadfile.loadImageSet('train-images.idx3-ubyte')  # import train set
labs1, head2 = loadfile.loadLabelSet('train-labels.idx1-ubyte')
for i in range(np.shape(labs1)[0]):  # change the labels to +1 and -1
    if labs1[i] == 0:
        labs1[i] = 1
    else:
        labs1[i] = -1

simplifysample1 = some_function.SimplifySample(500, imgs1)  # simplify sample by reducing features that are constant 0

# import data to be tested
if testSet == 0:
    # if testing train set
    imgs1_simplified = simplifysample1.remove_0_pixel_point(imgs1)
    test_imgs = imgs1_simplified[0:60000] / 255
    test_labs = labs1[0:60000] * 1
else:
    # if testing test set
    imgs2, head3 = loadfile.loadImageSet('t10k-images.idx3-ubyte')  # import test set
    labs2, head4 = loadfile.loadLabelSet('t10k-labels.idx1-ubyte')
    for i in range(np.shape(labs2)[0]):  # change the labels to +1 and -1
        if labs2[i] == 0:
            labs2[i] = 1
        else:
            labs2[i] = -1
    imgs2_simplified = simplifysample1.remove_0_pixel_point(imgs2)
    test_imgs = imgs2_simplified[0:10000] / 255
    test_labs = labs2[0:10000] * 1

M, N = np.shape(test_imgs)
test_imgs = np.c_[np.ones(M), test_imgs]
N = N + 1
test_imgs = np.mat(test_imgs)
test_labs = np.mat(test_labs).T

if construct_feature == 1:  # for the case of constructing extra features
    if linear == 1:  # if using linear LS method
        random_mat = np.loadtxt("p_classifier1_2Matrix_11_28_14_59.txt")
        x = np.mat(np.loadtxt("p_classifier1_2_11_28_14_59.txt")).T
    else:  # if using nonlinear LS method
        random_mat = np.loadtxt("p_classifier2_2Matrix_05_21_20_30.txt")
        x = np.mat(np.loadtxt("p_classifier2_2_05_21_20_30.txt")).T
    number_random_feature = np.shape(random_mat)[0]
    # Construct features matrix feature_mat
    random_feature_mat = test_imgs * random_mat.T  # construct random feature
    #   turn the negative elements in random_feature_matrix to 0
    artificial_feature_mat = (random_feature_mat + abs(random_feature_mat)) / 2
    feature_mat = np.hstack((test_imgs, artificial_feature_mat))
    N = N + number_random_feature  # number of features
else:   # for the case of constructing no extra features
    if linear == 1:
        feature_mat = test_imgs
        x = np.mat(np.loadtxt("p_classifier1_1_11_28_18_52.txt")).T
    else:
        feature_mat = test_imgs
        x = np.mat(np.loadtxt("p_classifier2_1_05_21_20_09.txt")).T

# calculate the result using linear LS method or nonlinear LS method
if linear == 1:
    A = feature_mat
    predict_labs = A * x
else:
    predict_labs = np.mat(np.ones(M) * 1.0).T  # nonlinear function's value
    for i in range(M):
        u = feature_mat[i] * x
        # calculate nonlinear function's value
        u = u[0, 0]
        predict_labs[i] = 1 - 2 / (e ** (2 * u) + 1)

# print the predict result
compare_result = np.hstack((test_labs, predict_labs))
print(compare_result)
print(N)

# calculate rightly  classified number and rightly classified rate
result = np.sign(predict_labs)
right_labs = np.abs(result + test_labs) * 0.5
number_of_right_labs = np.sum(right_labs)
right_rate = number_of_right_labs / M
print('\nclassification accuracy is: ', right_rate, '\n')

# the rate of classify 1 rightly
number1_0 = sum(test_labs + abs(test_labs))/2  # the number of 1 in test_labs
mid = (test_labs + result) / 2
number2_0 = sum(mid + abs(mid))/2  # the number of 1 that are rightly classified
print("the number of pictures labeled 1 in test_labs:", int(number1_0[0,0]))
print("the number of pictures labeled 1 in test_labs that are rightly classified:", int(number2_0[0,0]))
print("classification accuracy for pictures labeled 1 is: ", number2_0[0,0]/number1_0[0,0])

# the rate of classify -1 rightly
number1_1 = M - number1_0  # the number of -1 in test_labs
mid = (test_labs + result) / 2
number2_1 = sum(-mid + abs(mid))/2  # the number of -1 that are rightly classified
print("the number of pictures labeled -1 in test_labs:", int(number1_1[0,0]))
print("the number of pictures labeled -1 in test_labs that are rightly classified:", int(number2_1[0,0]))
print("classification accuracy for pictures labeled -1 is: ", number2_1[0,0]/number1_1[0,0])
