import time
import os
import matplotlib.pyplot as plt
import pandas as pd


from main_app import app_demo, MAX_N_MSGS, MAX_N_CLIENTS
from NetworkNode.utils import save_pickle, load_pickle
from NetworkNode import POOL_SIZE

import warnings
warnings.filterwarnings("ignore")

# N_CLIENT = [2 ** n for n in range(5, 13)]
N_CLIENT = [50, 100, 200, 400, 600, 800, 1000]
# N_CLIENT=[100]
N_layer=3
N_point_each_layer=3
N_RELAYS = N_layer*N_point_each_layer
DEFAULT_N_MSGS = 1
DEFAULT_N_CLIENTS = 16
# POOL_SIZES = [16, 32, 64, 128]


def evaluate_performance_wrt_n_clients():
    throughput_arr = []
    latency_arr = []
    time_cost = []
    init_times = []
    for n_clients in N_CLIENT:

        # run the app
        init_time,receive_time=app_demo(n_relays=N_RELAYS,n_layer=N_layer, n_clients=n_clients, n_msgs=DEFAULT_N_MSGS)

        # add the average to the throughput array
        init_times.append(init_time)
        time_cost.append(receive_time)
        # throughput_arr.append((DEFAULT_N_MSGS * n_clients) / (end - start))
        # latency_arr.append((end - start) / (DEFAULT_N_MSGS * n_clients))
        print("Test for this round ending........")

        time.sleep(2)
        # save_pickle(f'pkl/thr_n_clients_pool={POOL_SIZE}.pkl', throughput_arr)
    return time_cost, init_times


def plot_throughput(eval_func: callable, save=False):
    time_cost, init_times = eval_func()
    # fig, axis = plt.subplots(1, 2)
    #
    # axis[0].set_title(f'Throughput (pool size={POOL_SIZE})')
    # axis[0].set_xlabel('n_clients')
    # axis[0].set_ylabel('msgs/second')
    # axis[0].plot(N_CLIENT, thr_arr, marker='.', lw=1.5, color='orange')
    #
    # axis[1].set_title(f'Latency (pool size={POOL_SIZE})')
    # axis[1].set_xlabel('n_clients')
    # axis[1].set_ylabel('seconds')
    # axis[1].plot(N_CLIENT, lat_arr, marker='.', lw=1.5, color='purple')
    #
    # fig.show()

    # if save:
    #     i = 1
    #     filename = f'{eval_func.__name__}-pool={POOL_SIZE}-{{i}}'
    #     while os.path.exists(f'./png/{filename.format(i=i)}.png') and i < 100:
    #         i += 1
    #     filename = filename.format(i=i)
    #     fig.savefig(f'./png/{filename}.png')
    #     save_pickle(f'./pkl/{filename}-throughput.pkl', thr_arr)
    #     save_pickle(f'./pkl/{filename}-latency.pkl', lat_arr)

    df=pd.DataFrame({'candiate clients':N_CLIENT,'com_time_cost':time_cost,'init_time':init_times})
    df.to_excel("output.xlsx",index=False)

if __name__ == '__main__':
    plot_throughput(evaluate_performance_wrt_n_clients, save=True)
