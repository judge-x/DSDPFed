import sys
import threading
import time
from typing import List, Tuple

from NetworkNode import *

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 65432
RELAY_PREFIX_ADDR = '127.0.1.{i}'
CLIENT_PREFIX_ADDR = '127.0.2.{i}'


def setup_relays(n: int) -> List[Relay]:
    """
    setup n relays
    :param n: amount of relays to create
    :return: list of relays
    """
    relays = []
    print(f'setting up {n} relays...', end='')
    for i in range(1, n + 1):
        relays.append(Relay(f'127.0.1.{i}', DEFAULT_PORT))
    Relay.setup_relay_chain(relays)
    print('done')
    return relays


def setup_clients(n: int) -> List[Client]:
    """
    setup n clients
    :param n: amount of clients to create
    :return: list of n clients
    """
    clients = []
    print(f'setting up {n} clients...', end='')
    for i in range(1, n + 1):
        client = Client(f'127.0.2.{i}')
        # client.set_host_server(DEFAULT_HOST, DEFAULT_PORT)
        clients.append(client)
    print('done')
    return clients


def start_client(client: Client, n_msg: int) -> None:
    """
    starts a network client action
    :param client: network client
    :param n_msg: number of messages to send
    :return:
    """
    # call to sleep, so the os scheduler queue the client thread to run after all the relays are setup
    time.sleep(1)
    print(f'{client} sending message...\n')
    for i in range(n_msg):
        # core message
        core_msg = POST + f'msg-{i + 1} from: {client}'.encode() + END
        # core_msg = f'POST:hello'
        client.send_through_chain(DEFAULT_HOST, DEFAULT_PORT, core_msg)
        # create delay between sending messages
        time.sleep(1)
    print(f'{client} disconnecting.\n')


def init_network(n_clients: int, n_relays: int) -> Tuple[List[Client], List[Relay], Server]:
    """
    initialize the network components: server, mixnet (relays chain) and clients
    :param n_clients: amount of client to create
    :param n_relays: amount of relays to create
    :return: list of clients, list of relays, server
    """
    # setup relays and clients
    relays = setup_relays(n_relays)
    clients = setup_clients(n_clients)
    print(f'setting up server {DEFAULT_HOST}\n')
    server = Server(DEFAULT_HOST, DEFAULT_PORT)
    return clients, relays, server


def init_threads(clients: List[Client], relays: List[Relay], server: Server, n_msgs: int) -> None:
    """
    initialize the threads bound to the actions of the network components.
    for each network component, define the function that should be executed.
    :param clients: list of clients to create threads for
    :param relays: list of relays to create threads for
    :param server: server to create a thread for
    :param n_msgs: number of messages each client should send
    :return:
    """
    th_relays = []  # list of relays threads
    th_clients = []  # list for clients threads
    # setup threads for different relays and different clients, and the server
    th_server = threading.Thread(target=server.receive, args=([],), name=str(server))
    for client in clients:
        th_clients.append(threading.Thread(target=start_client, args=(client, n_msgs), name=str(client)))
        client.set_relays_chain(relays)
    for relay in relays:
        th_relays.append(threading.Thread(target=relay.receive, name=str(relay)))

    # start threads (client is going to sleep, in order to let the other threads work before it)
    th_server.start()
    for tr in th_relays:
        tr.start()
    for tc in th_clients:
        tc.start()

    # join all threads
    th_server.join()
    for tr in th_relays:
        tr.join()
    for tc in th_clients:
        tc.join()


def mixnet_demo():
    """"
    demo of the network components: clients, relays (mixnet) and server
    """
    # get user input parameters
    n_relays = int(sys.argv[1])
    n_clients = int(sys.argv[2])
    n_msgs = int(sys.argv[3])

    print(f'MixNet-DEMO information:'
          f'\n**************'
          f'\ndebug mode: {DEBUG_MODE}'
          f'\npool size: {POOL_SIZE}'
          f'\nrelays: {n_relays}'
          f'\nclients: {n_clients}'
          f'\neach client sends {n_msgs} messages'
          f'\nmsg size is: {MSG_MAX_SIZE}'
          f'\nsocket timeout is: {SOCKET_TIMEOUT}'
          f'\n**************\n')

    # start demo
    clients, relays, server = init_network(n_clients, n_relays)
    # setup server public key for onion encryption
    for client in clients:
        client.set_host_pb_key(server.get_public_key())
    init_threads(clients, relays, server, n_msgs)


if __name__ == '__main__':
    mixnet_demo()
