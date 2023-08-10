import argparse
import threading
import os

from main_app import app_demo, DEFAULT_PORT, DEFAULT_HOST, start_threads, join_threads, setup_client_app, MAX_N_CLIENTS, \
    MAX_N_MSGS
from App import *
from NetworkNode import Relay, SOCKET_TIMEOUT, POOL_SIZE
from NetworkNode.utils import load_key_pair

KEYS_DIR = './keys'

WAIT_TIME = 5
MSG_POOL_SIZE = f'MixNet pool size: {POOL_SIZE}'
MSG_SERVER_ADDRESS = f'server ip address:'
MSG_SERVER_PORT = f'server port:'



def simple_relays_setup():
    """
    setup 3 relays and their corresponding threads
    :return: list of relays, list of relays threads
    """
    relays = [Relay('127.1.0.1', DEFAULT_PORT),
              Relay('127.1.0.2', DEFAULT_PORT),
              Relay('127.1.0.3', DEFAULT_PORT)]
    Relay.setup_relay_chain(relays)
    th_relays = []
    for relay in relays:
        th_relays.append(threading.Thread(target=relay.receive, name=str(relay)))
    return relays, th_relays


def init_parser() -> argparse.ArgumentParser:
    """
    init the argument parser of the program
    :return: argument parser
    """
    parser = argparse.ArgumentParser(description='final project in course:'
                                                 'Advanced Topics in Online Privacy and Cyber-security (67515): '
                                                 'Anonymizing Public Transportation rides information')
    parser.add_argument('-d', '--demo-mode', action='store_true',
                        help='run demonstration mode. clients send random messages to the server. '
                             'this option executes app_demo in mot_app.py')
    parser.add_argument('-s', '--server', action='store_true',
                        help='setup a server to run on the machine. use -a to choose address for server. '
                             'use -p to choose port for server. otherwise, default ip address and port will be set.')
    parser.add_argument('-c', '--clients',
                        nargs=2, type=int, metavar=('n_clients', 'n_messages'),
                        help='setup n_clients to send n_messages to the server through the MixNet. '
                             'use -a to choose address for server. '
                             'use -p to choose port for server. '
                             'otherwise, default ip address and port will be set.')
    parser.add_argument('-p', '--port', type=int, metavar='server_port',
                        help='port number of the MoT server')
    parser.add_argument('-a', '--address', type=str, metavar='server_address',
                        help='ip address of the MoT server')

    return parser


def demo_mode():
    """
    start demo mode of program
    :return:
    """
    n_clients = 10
    n_relays = 3
    n_msgs = 1
    print(f'running demo mode...')
    app_demo(n_relays, n_clients, n_msgs)


def server_mode(server_ip_address: str, server_port: int):
    """
    start server mode of the program
    :param server_ip_address: ip address of server
    :param server_port: port number of server
    :return:
    """
    server_app = ServerApp(server_ip_address, server_port, name='MotApp')
    print('running server mode...'
          f'\n{MSG_SERVER_ADDRESS} {server_ip_address}'
          f'\n{MSG_SERVER_PORT} {server_port}'
          f'\n{MSG_POOL_SIZE}'
          f'\nsocket timeout: {SOCKET_TIMEOUT} seconds')
    # start threads without any clients threads and relay threads
    start_threads(server_app, [], [])
    join_threads(server_app, [], [])


def clients_mode(n_clients: int, n_msgs: int, server_address: str, server_port: int):
    """
    start clients mode of the program
    :param n_clients: number of client applications to create
    :param n_msgs: number of messages each client-app should send
    :param server_address: ip address of server bound to client app
    :param server_port: port number of server bound to client app
    :return:
    """
    n_clients = min([n_clients, MAX_N_CLIENTS])
    n_msgs = min([n_msgs, MAX_N_MSGS])
    print(f'running clients mode...'
          f'\nsetting up {n_clients} clients.'
          f'\neach client sends {n_msgs} message(s)'
          f'\n{MSG_SERVER_ADDRESS} {server_address}'
          f'\n{MSG_SERVER_PORT} {server_port}'
          f'\n{MSG_POOL_SIZE}')

    # print(f'waiting for {WAIT_TIME} seconds for user to set up server...', end='')
    # time.sleep(WAIT_TIME)
    # print('done')

    # setup relays and client apps
    relays, th_relays = simple_relays_setup()
    # get server public key
    server_pbkey = load_key_pair(('server_pr_key', 'server_pb_key'))[1]
    client_apps = setup_client_app(n_clients, relays, n_msgs, server_address, server_port, server_pbkey)
    # start and join the threads
    start_threads(None, client_apps, th_relays)
    join_threads(None, client_apps, th_relays)


def main():
    # init the argument parser and get arguments
    parser = init_parser()
    args = parser.parse_args()
    # setup server ip address and port information
    if args.address is not None:
        server_address = args.address
    else:
        server_address = DEFAULT_HOST
    if args.port is not None:
        server_port = args.port
    else:
        server_port = DEFAULT_PORT

    # run demo mode
    if args.demo_mode:
        demo_mode()
    # if clients flag given start program in clients mode
    elif args.clients is not None:
        n_clients, n_msgs = args.clients
        if n_clients <= 0 or n_msgs <= 0:
            raise ValueError('n_clients and n_msgs must be positive integers')
        clients_mode(n_clients, n_msgs, server_address, server_port)
    # in only the server flag was given, setup the server on the machine
    elif args.server:
        server_mode(server_address, server_port)
    # if no flags were given, print help instructions
    else:
        parser.print_help()


if __name__ == '__main__':
    # check that the program runs from root directory of the project
    if not os.path.exists('./App') or not os.path.exists('./NetworkNode'):
        raise Exception('the program must be executed from root folder')
    # check if keys directory is missing, and create it accordingly
    if not os.path.exists(KEYS_DIR):
        os.mkdir(KEYS_DIR)
    # run main
    main()
