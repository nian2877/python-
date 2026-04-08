import numpy as np


def sigmoid(z):
    """
    sigmoid 函数把任意实数映射到 (0, 1)。
    吴恩达课程里会把它当作逻辑回归的激活函数，
    用来表示“属于正类的概率”。
    """
    return 1 / (1 + np.exp(-z))


def initialize_parameters(dim):
    """
    初始化参数：
    w 的形状是 (特征数, 1)
    b 是一个标量
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    """
    前向传播 + 反向传播

    参数说明：
    X: 输入特征，形状 (特征数, 样本数)
    Y: 标签，形状 (1, 样本数)
    w: 权重，形状 (特征数, 1)
    b: 偏置，标量
    """
    m = X.shape[1]

    # 前向传播：Z = w^T X + b
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    # 交叉熵损失
    cost = -1 / m * np.sum(Y * np.log(A + 1e-10) + (1 - Y) * np.log(1 - A + 1e-10))

    # 反向传播：对 w 和 b 求导
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    grads = {"dw": dw, "db": db}
    return grads, float(cost)


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    使用梯度下降不断更新参数
    """
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        # 梯度下降更新公式
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"第 {i} 次迭代，损失值: {cost:.6f}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs


def predict(w, b, X):
    """
    根据训练好的参数进行预测
    概率 >= 0.5 预测为 1，否则预测为 0
    """
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    Y_prediction = (A >= 0.5).astype(int)
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.1, print_cost=False):
    """
    逻辑回归整体模型流程：
    1. 初始化参数
    2. 梯度下降学习参数
    3. 用训练好的参数做预测
    """
    w, b = initialize_parameters(X_train.shape[0])

    params, grads, costs = optimize(
        w,
        b,
        X_train,
        Y_train,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        print_cost=print_cost,
    )

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

    print(f"训练集准确率: {train_accuracy:.2f}%")
    print(f"测试集准确率: {test_accuracy:.2f}%")

    result = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }
    return result


if __name__ == "__main__":
    # 这里构造一个非常小的二维样本集，方便你理解逻辑回归在做什么。
    # 每一列代表一个样本，每一行代表一个特征。
    X_train = np.array([
        [0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5],
        [3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.5, 0.2],
    ])
    Y_train = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])

    X_test = np.array([
        [1.2, 2.8, 3.8],
        [2.4, 1.1, 0.4],
    ])
    Y_test = np.array([[0, 1, 1]])

    print("开始训练逻辑回归模型...")
    model(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_iterations=1500,
        learning_rate=0.05,
        print_cost=True,
    )
