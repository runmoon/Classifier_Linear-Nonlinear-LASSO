import numpy as np
import loadfile
from math import *
import time
import some_function

# Train-----------------------------------------------------------------------------------------------------------------
construct_feature = 1  # if is set 1, means construct some extra random features; if is set 0, not
np.random.seed(5)

imgs1, head1 = loadfile.loadImageSet('train-images.idx3-ubyte')  # import train set
labs1, head2 = loadfile.loadLabelSet('train-labels.idx1-ubyte')
for i in range(np.shape(labs1)[0]):
    if labs1[i] == 0:
        labs1[i] = 1
    else:
        labs1[i] = -1
M, N = np.shape(imgs1);
print("M = ", M)
print("N = ", N)

simplifysample1 = some_function.SimplifySample(500, imgs1)
imgs1_simplified = simplifysample1.remove_0_pixel_point(imgs1)  # simplify sample by reducing invalid features that are constant 0
M, N = np.shape(imgs1_simplified)
print("M = ", M)
print("N = ", N)

denominator = 255  # normalize every element in feature matrix to 0-1
train_imgs = imgs1_simplified[0:60000] / denominator
train_labs = labs1[0:60000] * 1

M, N = np.shape(train_imgs)
train_imgs = np.c_[np.ones(M), train_imgs]
train_imgs = np.mat(train_imgs)
N = N + 1  # number of features
train_labs = np.mat(train_labs).T
print("M = ", M)
print("number of features = ", N)

if construct_feature == 1:
    # Construct features matrix feature_mat
    number_random_feature = 500
    random_mat = np.random.randint(0, 1, number_random_feature * N) * 2 - 1  # generate random matrix consist of -1 or 1
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

print("number of final features = ", N)

lambda1 = 10  # 10 the coefficient that set to minimum the difference of consecutive iteration variable (x2 - x1)
lambda2 = 0  # regularization parameter

I1 = np.mat(np.eye(N) * 1)
I0 = np.mat(np.eye(N) * 1)
I0[N - 1, N - 1] = 0
I0 = np.mat(I0)
# x1 or x2 is the parameters we want to learn; or the variables that can minimum the objective function
x1 = np.mat(np.random.rand(N) * 2 - 1).T * 0.1  # initializing parameters by generating random number between among[-1,-1];
x2 = np.mat(np.random.rand(N) * 2 - 1).T * 0.1
if construct_feature == 1:
    # random_feature_initial = (np.random.rand(number_random_feature) * 2 - 1) * 0.1
    # x1 = np.r_[np.loadtxt('record_classifier2_1.txt'), random_feature_initial]  # initializing parameters by previous result when constructing extra random features;
    x1 = np.mat(np.random.rand(N) * 2 - 1).T * 0.1
    x2 = np.mat(np.random.rand(N) * 2 - 1).T * 0.1
else:
    x1 = np.mat(np.random.rand(N) * 2 - 1).T * 0.1  # initializing parameters by generating random number between among[-1,-1];
    x2 = np.mat(np.random.rand(N) * 2 - 1).T * 0.1

f = np.mat(np.ones(M) * 1.0).T  # nonlinear function's value
Df = np.mat(np.ones([M, N]) * 1.0)  # Jacobian matrix of f
e1, e2, e3 = [M * 10 ** (-6), N * 10 ** (-6), M * 10 ** (-6)]  # parameters for termination determining
sum_f1 = 0  # second-order norm of vector-valued function f
sum_f2 = 0

iteration = 0
iteration_upper_limit = 10
while iteration < iteration_upper_limit:
    iteration = iteration + 1
    print('iteration = ', iteration)

    sum_f2 = 0
    # construct: approximate function--f; and Jacobian matrix of f--Df
    for i in range(M):
        u = feature_mat[i] * x1
        # calculate nonlinear function's value and derivative for sample imgs[i]
        e_2u = e**(2*u[0, 0])
        f[i] = 1 - 2/(e_2u + 1) - train_labs[i, 0]
        f_diff = 4 * e_2u / (e_2u + 1)**2
        sum_f2 = sum_f2 + f[i]**2
        # Jacobian matrix of sample train_imgs[i]
        Df[i] = f_diff * feature_mat[i]
    # update x2
    # A = Df.T * Df + lambda1 * I1 + lambda2 * I0
    # b = Df.T * f + lambda2 * I0 * x1
    # Q, R = np.linalg.qr(A)
    # x2 = x1 - np.linalg.inv(R) * Q.T * b
    x2 = x1 - np.mat((Df.T * Df + lambda1 * I1 + lambda2 * I0)).I * (Df.T * f + lambda2 * I0 * x1)

    if 1==iteration:
        sum_f1 = 2*sum_f2
    print('sum_f1 = ', sum_f1)
    print('sum_f2 = ', sum_f2)

    # termination determine
    d_x = x2 - x1
    sum_d_x = 0
    Df_f = Df.T * f
    sum_Df_f = 0
    for i in range(N):
        sum_d_x = sum_d_x + d_x[i]**2
        sum_Df_f = sum_Df_f + Df_f[i]**2
    # if ||f|| or ||x2-x1|| or||Df*f|| is small enough, terminate!
    if np.sqrt(sum_f2) < e1:
        print("迭代结束，sum_f1 足够小了！")
        break
    if np.sqrt(sum_d_x) < e2:
        print("迭代结束，sum_d_x 足够小了！")
        break
    if np.sqrt(sum_Df_f) < e3:
        print("迭代结束，sum_Df_f 足够小了！的x值")
        break
    # determine whether accept the x2; and update p1
    if sum_f2 < sum_f1:  # accept the x2
        print("接受新的x值")
        x1 = x2
        lambda1 = 0.8 * lambda1  # update p1
        sum_f1 = sum_f2
    else:  # don't accept the x2
        print("不接受新的x值")
        lambda1 = 2 * lambda1

x = x2



# Test-----------------------------------------------------------------------------------------------------------------
imgs2, head3 = loadfile.loadImageSet('t10k-images.idx3-ubyte')
labs2, head4 = loadfile.loadLabelSet('t10k-labels.idx1-ubyte')
for i in range(np.shape(labs2)[0]):
    if labs2[i] == 0:
        labs2[i] = 1
    else:
        labs2[i] = -1

# imgs2_simplified = some_function.remove_0_pixel_point( 600, imgs2 )
# test_imgs = imgs2[0:10000] / denominator
# test_labs = labs2[0:10000] * 1
test_imgs = imgs1_simplified[0:60000] / denominator
test_labs = labs1[0:60000] * 1

M, N = np.shape(test_imgs)
test_imgs = np.c_[np.ones(M), test_imgs]
N = N + 1
test_imgs = np.mat(test_imgs)
test_labs = np.mat(test_labs).T

if construct_feature == 1:
    # Construct features matrix feature_mat
    random_feature_mat = test_imgs * random_mat.T  # construct random feature
    #   turn the negative elements in random_feature_matrix to 0
    artificial_feature_mat = (random_feature_mat + abs(random_feature_mat)) / 2
    feature_mat = np.hstack((test_imgs, artificial_feature_mat))
    N = N + number_random_feature  # number of features
else:
    feature_mat = test_imgs

predict_labs = np.mat(np.ones(M) * 1.0).T  # nonlinear function's value
for i in range(M):
    u = feature_mat[i] * x
    # calculate nonlinear function's value
    u = u[0, 0]
    predict_labs[i] = 1 - 2 / (e ** (2 * u) + 1)


# print result----------------------------------------------------------------------------------------------------------
# print the predict result
compare_result = np.hstack((test_labs, predict_labs))
print(compare_result[0:200])

# calculate rightly classified number and rightly classified rate
result = np.sign(predict_labs)
right_labs = np.abs(result + test_labs) / 2  # rightly classified number
number_of_right_labs = np.sum(right_labs)
right_rate = number_of_right_labs / M  # rightly classified rate
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
number2_1 = number_of_right_labs - number2_0  # the number of -1 that are rightly classified
print("the number of pictures labeled -1 in test_labs:", int(number1_1[0,0]))
print("the number of pictures labeled -1 in test_labs that are rightly classified:", int(number2_1[0,0]))
print("classification accuracy for pictures labeled -1 is: ", number2_1[0,0]/number1_1[0,0])

# save the training result to txt files
now_time = str(time.strftime('%m_%d_%H_%M', time.localtime(time.time())))
if construct_feature == 1:
    np.savetxt("p_classifier2_2_" + now_time + ".txt", x)  # save the learned parameters
    np.savetxt("p_classifier2_2Matrix_" + now_time + ".txt", random_mat)  # save the learned parameters
else:
    np.savetxt("p_classifier2_1_" + now_time + ".txt", x)  # save the learned parameters
