import numpy as np

def itemCF(user_item, user_id, N=10):
    # 计算物品之间的皮尔森相关系数
    item_sim = np.corrcoef(user_item.T)
    
    # 找到用户购买过的所有物品
    items_bought = np.where(user_item[user_id] > 0)[0]

    # 计算每个物品的得分
    scores = np.zeros(user_item.shape[1])
    for item in range(user_item.shape[1]):
        if item not in items_bought:
            # 计算物品item与用户已购买的物品之间的相似度加权平均得分
            sim_sum = 0
            score_sum = 0
            for bought_item in items_bought:
                if item_sim[item, bought_item] > 0:
                    sim_sum += item_sim[item, bought_item]
                    score_sum += item_sim[item, bought_item] * user_item[user_id, bought_item]
            if sim_sum == 0:
                scores[item] = 0
            else:
                scores[item] = score_sum / sim_sum
    
    # 对得分进行排序，取前N个作为推荐列表
    item_ranking = np.argsort(scores)[::-1][:N]
    
    return item_ranking, scores[item_ranking]

user_item = np.random.randint(6, size=(10, 20))

item_ranking, scores = itemCF(user_item, 2, N=5)

print("推荐的物品列表：", item_ranking)
print("推荐的物品得分：", scores)