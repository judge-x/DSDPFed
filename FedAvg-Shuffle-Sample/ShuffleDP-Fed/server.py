def update(shuffled_local_parameters,global_paramerter,init_parameter,num_client,lr):
    '''
        全局模型梯度更新，算法FedAvg,传统算法
    '''
    sum_parameter=init_parameter
    for x in shuffled_local_parameters:
        if sum_parameter is None:  # 更新全局的参数
            sum_parameter = {}
            for key, var in x.items():
                sum_parameter[key] = var.clone()
        else:
            for var in sum_parameter:
                sum_parameter[var] = (sum_parameter[var] + x[var] + global_paramerter[var])

    for var in sum_parameter:  # 加权平均赋值给全局变量
        sum_parameter[var] = sum_parameter[var] / len(shuffled_local_parameters)

    return sum_parameter
    # '''
    #     使用行的下降的梯度算法
    # '''
    # sum_parameter = init_parameter
    # for x in shuffled_local_parameters:
    #     if sum_parameter is None:  # 更新全局的参数
    #         sum_parameter = {}
    #         for key, var in x.items():
    #             sum_parameter[key] = var.clone()
    #     else:
    #         for var in sum_parameter:
    #             sum_parameter[var] += x[var]
    # for var in global_paramerter:
    #     global_paramerter[var]+=sum_parameter[var]*lr
    #
    # return global_paramerter
    #
    # '''
    # 使用连接截至的下降梯度算法
    # '''
    # sum_parameter = init_parameter
    # for x in shuffled_local_parameters:
    #     if sum_parameter is None:  # 更新全局的参数
    #         sum_parameter = {}
    #         for key, var in x.items():
    #             sum_parameter[key] = var.clone()
    #     else:
    #         for var in sum_parameter:
    #             sum_parameter[var] += x[var]/num_client
    # for var in global_paramerter:
    #     global_paramerter[var] += sum_parameter[var] * lr
    #
    # return global_paramerter
