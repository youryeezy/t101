import numpy
import numpy as np
import matplotlib.pyplot as plt
from time import time


def generate_linear(a, b, noise, filename, size=100):  #generate x,y for linear
    print('Generating random data y = a*x + b')
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b + noise * a * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')
    return(x, y)


def linear_regression_numpy(filename):  #linear regression with polyfit
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    print(np.shape(x))
    print(np.shape(y))

    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")

    h = model[0]*x + model[1]

    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return model


def linear_regression_exact(filename):  #custom linear regression
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    time_start = time()
    tmp_x = np.hstack([np.ones((100, 1)), x])
    trans_x = np.transpose(tmp_x)
    res_thetha = np.linalg.matrix_power(trans_x.dot(tmp_x), -1).dot(trans_x).dot(y)
    print(res_thetha)
    print(np.shape(x))
    print(np.shape(y))

    time_end = time()

    h = res_thetha[1] * x + res_thetha[0]
    print(f"Linear regression time:{time_end-time_start}")
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return res_thetha
    

def check(model, ground_truth):
    if len(model) != len(ground_truth):
        print("Model is inconsistent")
        return False
    else:
        r = np.dot(model-ground_truth, model-ground_truth)/(np.dot(ground_truth, ground_truth))
        print(r)
        if r < 0.0005:
            return True
        else:
            return False


def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    print(np.shape(x))
    print(np.shape(y))
    if len(a) != (n+1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0, n+1):
        y = y + a[i] * np.power(x, i) + noise*(np.random.rand(size, 1) - 0.5)
    print(np.shape(x))
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')


def polynomial_regression_numpy(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    print(np.shape(x))
    print(np.shape(y))
    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 2)
    time_end = time()
    print(f"Polinomial regression with polyfit in {time_end - time_start} seconds")
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    x = np.sort(x, axis=0)
    h = model[0] * x * x + model[1] * x + model[2]

    plt.plot(x, h, "r", label='model')

    plt.legend()
    plt.show()
    return model


def gradient_descent_step(dJ, theta, alpha):
    theta_new = theta - alpha * 1/60 * dJ
    return theta_new


def get_dJ(x, y, theta):
    h = theta.dot(x.transpose())
    dJ = (h - y).dot(x)
    return dJ


def get_dJ_minibatch(x, y, theta):
    h = theta.dot(x.transpose())
    dJ = (h - y).dot(x)
    return dJ


def get_dJ_sgd(x, y, theta):
    h = theta.dot(x.transpose())
    dJ = (h - y).dot(x)
    return dJ



def minimize(x, y, L):
    # n - number of samples in learning subset, m - ...
    n = 2  # <-- calculate it properly!
    theta = np.ones(n)  # you can try random initialization
    # dJ = np.zeros(n)
    for i in range(0, L):
        theta = get_dJ(x, y, theta)  # here you should try different gradient descents
        J = 0  # here you should calculate it properly
    # and plot J(i)
    print("your code goes here")
    return


if __name__ == "__main__":
    generate_linear(1, -3, 1, 'linear.csv', 100)
    model = np.squeeze(linear_regression_exact("linear.csv"))
    poly_model = polynomial_regression_numpy("polynomial.csv")
    # print(f"Is model correct?\n{check(model, np.array([1,-3]))}")
    mod1 = np.squeeze(numpy.asarray(np.array(([-3], [1]))))
    print(f"Is model correct?\n{check(model, mod1)}")

    # ex1 . - exact solution
    # model_exact = linear_regression_exact("linear.csv")
    # check(model_exact, np.array([-3,1]))
    
    # ex1. polynomial with numpy
    generate_poly([1, 2, 3], 2, 0.5, 'polynomial.csv')
    # polynomial_regression_numpy("polynomial.csv")

    # ex2. find minimum with gradient descent
    # 0. generate date with function above
    generate_linear(1, -3, 1, 'linear.csv', 100)
    # 1. shuffle data into train - test - valid
    with open('linear.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    train_data = data[:60]
    train_x, train_y = np.hsplit(train_data, 2)
    test_data = data[60::1]
    valid_data = data[80::1]
    # 2. call minuimize(...) and plot J(i)
    minimize(train_x, train_y, 10)
    # 3. call check(theta1, theta2) to check results for optimal theta
    
    # ex3. polinomial regression
    # 0. generate date with function generate_poly for degree=3, use size = 10, 20, 30, ... 100
    # for each size:
    # 1. shuffle data into train - test - valid
    # Now we're going to try different degrees of model to aproximate our data, set degree=1 (linear regression)
    # 2. call minimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta
    # 4. plot min(J_train), min(J_test) vs size: is it overfit or underfit?
    # 
    # repeat 0-4 for degres = 2,3,4

    # ex3* the same with regularization
    