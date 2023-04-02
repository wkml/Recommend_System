import numpy as np

# 定义模型超参数
num_users = 5
num_items = 4
latent_dimension = 3
learning_rate = 0.01
num_epochs = 1000

# 创建用户和物品的评分矩阵
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

# 初始化用户和物品的潜在因素矩阵
U = np.random.rand(num_users, latent_dimension)
V = np.random.rand(num_items, latent_dimension)

# 迭代训练模型
for epoch in range(num_epochs):
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:
                error = R[i, j] - np.dot(U[i], V[j])
                U[i] += learning_rate * (error * V[j])
                V[j] += learning_rate * (error * U[i])

# 计算预测评分矩阵
R_pred = np.dot(U, V.T)

# 输出预测评分矩阵
print(R_pred)
