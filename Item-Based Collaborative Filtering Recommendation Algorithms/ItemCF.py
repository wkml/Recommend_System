import numpy as np

# 载入数据
ratings = np.array([[3, 7, 4, 9, 8], 
                    [8, 5, 6, 4, 10], 
                    [7, 8, 5, 9, 7], 
                    [6, 4, 9, 7, 8], 
                    [7, 5, 8, 9, 6]])
user_id = 0

# Item-Based Collaborative Filtering
def item_based_cf(ratings, user_id):
    # 计算每个物品之间的相似度
    item_similarities = np.dot(ratings.T, ratings) / \
                        (np.sqrt(np.sum(ratings**2, axis=0)).reshape(-1,1) * np.sqrt(np.sum(ratings**2, axis=0)))

    # 计算目标用户没有评分的物品的预测评分
    pred_ratings = np.dot(ratings[:, :], item_similarities) / np.sum(item_similarities, axis=0)

    # 返回推荐列表
    sorted_indices = np.argsort(pred_ratings)[::-1]
    
    recommended_items = sorted_indices[np.where(ratings[user_id, sorted_indices] == 0)[0]]

    return recommended_items

# 输出推荐列表
print(item_based_cf(ratings, user_id))
