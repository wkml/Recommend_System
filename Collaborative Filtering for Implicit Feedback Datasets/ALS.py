import numpy as np

# 载入数据
ratings = np.array([[0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 1], 
                    [0, 0, 1, 1, 0], 
                    [1, 1, 0, 0, 1], 
                    [0, 1, 0, 1, 0]])

# 初始化参数
num_users, num_items = ratings.shape
latent_dimension = 2
lambda_val = 0.1
num_iterations = 10


def als(ratings, latent_dimension, lambda_val, num_iterations):
    # 初始化用户和物品的矩阵
    user_matrix = np.random.rand(num_users, latent_dimension)
    item_matrix = np.random.rand(num_items, latent_dimension)

    # 迭代学习用户和物品的矩阵
    for i in range(num_iterations):
        # 交替学习用户矩阵和物品矩阵
        item_matrix = np.linalg.solve(np.dot(user_matrix.T, user_matrix) + lambda_val * np.eye(latent_dimension), np.dot(user_matrix.T, ratings.T)).T
        user_matrix = np.linalg.solve(np.dot(item_matrix.T, item_matrix) + lambda_val * np.eye(latent_dimension), np.dot(item_matrix.T, ratings)).T

    # 返回学习得到的用户和物品的矩阵
    return user_matrix, item_matrix

# 调用ALS算法并输出结果
user_matrix, item_matrix = als(ratings, latent_dimension, lambda_val, num_iterations)
print("User matrix:\n", user_matrix)
print("Item matrix:\n", item_matrix)
