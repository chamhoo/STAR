from hyperopt import hp, fmin, tpe, Trials



# Hyper Space
# model_Hparameters contains 3 Hyper-parameters:
#     - n_layers: # of layers in the temporal encoder
#     - ratio: The number of mambas used by the spatial layer is multiple times more than that used by the temporal layer
#     - embedding

space = {
    'param1': hp.uniform('param1', 0, 1),  
    'param2': hp.choice('param2', ['option1', 'option2', 'option3']),  }








if __name__ == "__main__":
    trials = Trials()
    best = fmin(
        fn=objective_function,  # 目标函数
        space=space,  # 参数空间
        algo=tpe.suggest,  # 优化算法，这里使用TPE
        max_evals=100,  # 最大评估次数
        trials=trials  # 保存试验结果
    )