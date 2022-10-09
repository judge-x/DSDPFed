import random
def shuffle(local_parameters):
    '''
        shuffle scrambling algorithm
    '''
    local_parameters=local_parameters
    r = random.random
    random.seed(random.randint(0, 10))
    random.shuffle(local_parameters, random=r)