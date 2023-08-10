import threading
import sys
import time
from typing import List

from NetworkNode import *
from App import *

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 65432

SERVER_SUBNET = '127.0.{b3}.{b4}'
RELAY_SUBNET = '127.1.{b3}.{b4}'
CLIENT_SUBNET = '127.2.{b3}.{b4}'

MAX_N_CLIENTS = 20000
MAX_N_RELAYS = 16
MAX_N_MSGS = 128
MAX_ADDRESS_LSB = 255
N_MSGS_DEMO = 2


def compute_ip_address(address: str, byte3: int, byte4: int):
    if byte3 <= MAX_ADDRESS_LSB:
        return address.format(b3=byte3, b4=byte4)
    else:
        return address.format(b3=MAX_ADDRESS_LSB, b4=byte4)


def setup_server_app(address: str , port: int , n_message:int):
    if address is None:
        for byte3 in range(256):
            for byte4 in (1, 256):
                try:
                    # setup server app
                    ip_address = compute_ip_address(SERVER_SUBNET, byte3, byte4)
                    return ServerApp(ip_address, DEFAULT_PORT, name='MainAPP',n_mesg=n_message)
                except OSError:
                    continue
        # otherwise, raise an exception if did not find an appropriate ip address for the server
        raise OSError('could not setup server')
    else:
        return ServerApp(address, port, name='MainAPP')


def setup_relays(n_relays: int):
    relays_amount = min([n_relays, MAX_N_RELAYS])
    print(f'setting up {relays_amount} relays...', end='')
    relays = []  # list of relays instances
    th_relays = []  # list of relays threads
    if relays_amount==0:
        return None,None
    else:
        for byte3 in range(256):
            for byte4 in range(1, 256):
                try:
                    # setup ip address for relay
                    ip_address = compute_ip_address(RELAY_SUBNET, byte3, byte4)
                    # setup relay
                    relay = Relay(ip_address, DEFAULT_PORT)
                    relays.append(relay)
                    # setup relay thread
                    th_relays.append(threading.Thread(target=relay.receive, name=str(relay)))
                    # setup the relay chain if could create enough relays
                    if len(relays) == relays_amount:
                        # Relay.setup_relay_chain(relays)
                        print('done')
                        return relays, th_relays
                except OSError:
                    continue
    # otherwise, raise an exception
    raise OSError('could not setup relays chain')


def setup_client_app(n_clients: int,n_layer: int, relays: List[Relay], n_msgs: int, server_address, server_port, server_pbkey):
    # take the minimal value between the maximal allowed number of clients, and the given number of clients
    clients_amount = min([n_clients, MAX_N_CLIENTS])
    print(f'setting {clients_amount} clientApps...', end='')
    client_apps = []
    for byte3 in range(256):
        for byte4 in range(1, 256):
            try:
                ip_address = compute_ip_address(CLIENT_SUBNET, byte3, byte4)
                capp = ClientApp(ip_address,
                                 relays,
                                 n_layer,
                                 server_address,
                                 server_port,
                                 server_pbkey,
                                 n_msgs)
                client_apps.append(capp)
                # setup the relay chain if could create enough relays
                if len(client_apps) == clients_amount:
                    print('done')
                    return client_apps
            except OSError:
                continue
    # otherwise, raise an exception
    raise OSError('could not setup clients')


def start_threads(server_app, client_apps, th_relays):
    if server_app is not None:
        server_app.start_app()
    for c_app in client_apps:
        c_app.start_app()

    for tr in th_relays:
        tr.start()



def join_threads(server_app, client_apps, th_relays):
    # join all threads
    if server_app is not None:
        server_app.join_app()

    for c_app in client_apps:
        c_app.join_app()

    for tr in th_relays:
        tr.join()








def app_demo(n_relays, n_layer,n_clients, n_msgs: int):
    intt_start_time=time.time()
    n_relays = min([n_relays, MAX_N_RELAYS])
    n_clients = min([n_clients, MAX_N_CLIENTS])
    n_msgs = min([n_msgs, MAX_N_MSGS])
    print(f'APP-DEMO information:'
          f'\n**************'
          f'\ndata is encrypted: {not DEBUG_MODE}'
          f'\npool size: {POOL_SIZE}'
          f'\nrelays: {n_relays}'
          f'\nclients: {n_clients}'
          f'\neach client sends: {n_msgs} messages'
          f'\nmsg max size is: {MSG_MAX_SIZE}'
          f'\n**************\n')

    # setup relays infrastructure for the network
    relays, thd_relays = setup_relays(n_relays)
    # setup server app
    server_app = setup_server_app(None,None,n_clients)
    # set up client applications
    clients_apps = setup_client_app(n_clients,n_layer, relays, n_msgs,
                                    server_app.server.get_ip_address(),
                                    server_app.server.get_port(),
                                    server_app.server.get_public_key())

    init_end_time=time.time()
    init_time=init_end_time-intt_start_time
    print(f'init time is: ',init_time,'\n')


    # start all threads
    start_threads(server_app, clients_apps, thd_relays)
    # join all entities
    join_threads(server_app, clients_apps, thd_relays)
    receive_time=server_app.server._end_time-server_app.server._start_time
    print('received all meassages in:',receive_time)
    server_app.server.close_socket()
    # print(clients_apps[0].get_rides_history())
    return init_time,receive_time


if __name__ == '__main__':
    # usr_input = input('generate rides example file? (y/n)')
    # if usr_input.lower() == 'y':
    #     generate_rides_example_file()
    #     exit(0)
    usr_input = input('run main-app-demo? (y/n)')
    if usr_input.lower() == 'y':
        n_relays = 3
        n_clients = 32
        # run demo
        app_demo(n_relays, n_clients, N_MSGS_DEMO)
