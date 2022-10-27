# LR2

### Линейная регрессия, полиномиальная регрессия, метод градиентного спуска

Рекомендуется использоваться Ubuntu 20.04 с установленными:
- python3
- numpy
- matplotlib
- sklearn

**Цель** Реализация и оптимизация метода градиентного спуска, решение задач регрессии

**Учебные задачи**
- освоить базовые операции _numpy_ (импорт/экспорт numpy array, перемножение матриц)
- решение задач регрессии с помощью _polyfit_
- освоить построение графиков в _matplotlib_

**Задачи**
1. Реализовано точное решение задачи поиска решения задачи регрессии

```python
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
    
```

2. Полиномиальная регрессия с помощью _numpy polyfit_

```python
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
```

3. Реализован метод градиентного спуска

```python
# Ex.3 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i (shape is 1 x N - the same as theta)
# x and y are both vectors

def gradient_descent_step(dJ, theta, alpha):
    theta_new = theta - alpha * 1/60 * dJ
    return theta_new


def get_dJ(x, y, theta):
    h = theta.dot(x.transpose())
    dJ = (h - y).dot(x)
    return dJ


def minimize(x, y, L):
    alpha = 0.15
    # n - number of samples in learning subset, m - ...
    n = 2  # <-- calculate it properly!
    theta = np.ones((1, n))  # you can try random initialization
    for i in range(0, L):
        dJ = get_dJ(x, y, theta)  # here you should try different gradient descents
        theta = gradient_descent_step(dJ, theta, alpha)
        alpha -= 0.0002
        h = theta.dot(x.transpose())
        J = 1 / 144 * (np.square(h - y)).sum(axis=1)  # here you should calculate it properly
        plt.plot(i, J, "b.")
    plt.legend()
    plt.show()
    return


def minimize_minibatch(x, y, L, M):  # M-size minibatch
    alpha = 0.15
    n = 2  # <-- calculate it properly!
    theta = np.ones((1, n))  # you can try random initialization
    x = np.vsplit(x, np.shape(x)[0] / M)
    y = np.hsplit(y, np.shape(y)[1] / M)
    for i in range(0, L):
        for x_minib, y_minib in list(zip(x, y)):
            dJ = get_dJ(x_minib, y_minib, theta)  # here you should try different gradient descents
            theta = gradient_descent_step(dJ, theta, alpha)
            alpha -= 0.00002
            h = theta.dot(x_minib.transpose())
            J = 1 / 144 * (np.square(h - y_minib)).sum(axis=1)  # here you should calculate it properly
        plt.plot(i, J, "b.")
    plt.legend()
    plt.show()
    return theta


def minimize_sgd(x, y, L):
    alpha = 0.30
    n = 2  # <-- calculate it properly!
    theta = np.ones((1, n))  # you can try random initialization
    for iter in range(0, L):
        for i, line in enumerate(x):
            one_y = np.reshape(y[0][i], (1, 1))
            line = line.reshape((1, 2))
            dJ = get_dJ(line, one_y, theta)  # here you should try different gradient descents
            theta = gradient_descent_step(dJ, theta, alpha)
            alpha -= 0.00002
            h = theta.dot(line.transpose())
            J = 1/144 * (np.square(h - one_y))  # here you should calculate it properly
        plt.plot(iter, J, "b.")
    plt.legend()
    plt.show()
    return theta
```


