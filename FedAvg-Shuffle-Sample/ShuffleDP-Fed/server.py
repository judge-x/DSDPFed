def update(shuffled_local_parameters,global_paramerter,init_parameter,num_client,lr):
    '''
        Global model gradient update, algorithm FedAvg, conventional algorithm
    '''
    sum_parameter=init_parameter
    for x in shuffled_local_parameters:
        if sum_parameter is None:
            sum_parameter = {}
            for key, var in x.items():
                sum_parameter[key] = var.clone()
        else:
            for var in sum_parameter:
                sum_parameter[var] = (sum_parameter[var] + x[var] + global_paramerter[var])

    for var in sum_parameter:
        sum_parameter[var] = sum_parameter[var] / len(shuffled_local_parameters)

    return sum_parameter
