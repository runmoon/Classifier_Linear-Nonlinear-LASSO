import numpy as np
import loadfile
import time
import some_function

# train----------------------------------------------------------------------
construct_feature = 1  # if is set 1, means construct some extra random features; if is set 0, not

imgs1, head1 = loadfile.loadImageSet('train-images.idx3-ubyte')  # import train set
labs1, head2 = loadfile.loadLabelSet('train-labels.idx1-ubyte')
for i in range(np.shape(labs1)[0]):
    if labs1[i] == 0:
        labs1[i] = 1
    else:
        labs1[i] = -1

simplifysample1 = some_function.SimplifySample(600, imgs1)
imgs1_simplified = simplifysample1.remove_0_pixel_point(imgs1)  # simplify sample by reducing features that are constant 0

train_imgs = imgs1_simplified[0:60000] / 255
train_labs = labs1[0:60000] * 1

M, N = np.shape(train_imgs)
train_imgs = np.c_[np.ones(M), train_imgs]
train_imgs = np.mat(train_imgs)
N = N + 1  # number of features
train_labs = np.mat(train_labs).T

if construct_feature == 1:
    # Construct features matrix feature_mat
    number_random_feature = 5000
    random_mat = np.random.randint(0, 2, number_random_feature * N) * 2 - 1  # generate random matrix consist of -1 or 1
    random_mat = np.reshape(random_mat, [number_random_feature, N])
    random_mat = np.mat(random_mat)
    random_feature_mat = train_imgs * random_mat.T
    # construct artificial features by function max{0, random_feature_mat}
    #   turn the negative elements in random_feature_mat to 0
    artificial_feature_mat = (random_feature_mat + abs(random_feature_mat)) / 2

    feature_mat = np.hstack((train_imgs, artificial_feature_mat))
    N = N + number_random_feature  # number of features
else:
    feature_mat = train_imgs

# solve equation
A = feature_mat
b = train_labs
Q, R = np.linalg.qr(A)
x = np.linalg.pinv(R) * Q.T * b  # solution

# save the training result to txt files
now_time = str(time.strftime('%m_%d_%H_%M',time.localtime(time.time())))
if construct_feature == 1:
    np.savetxt("p_classifier1_2_" + now_time + ".txt", x)  # save the learned parameters
    np.savetxt("p_classifier1_2Matrix_" + now_time + ".txt", random_mat)  # save the learned parameters
else:
    np.savetxt("p_classifier1_1_" + now_time + ".txt", x)  # save the learned parameters
