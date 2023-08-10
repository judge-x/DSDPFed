import pickle
import random
from typing import List
import threading
import time
import pandas as pd

from NetworkNode import *
from .message_app import MotMessage, ride_generator, LINE_NUMBER, OPERATOR, BOARDING_TIME, \
    STATION_DEST, STATION_SOURCE, TRAVEL_CODE, COLS


class ClientApp:
    """
    represents the client-side application
    """

    def __init__(self, client_address: str, relays: List[Relay], n_layer: int,
                 host: str, port: int, host_pb_key=None,
                 n_msgs: int = 1) -> None:
        """
        init a client-application instance
        :param client_address: ip address of client
        :param relays: list of relays making the chain in the mixnet
        :param host: ip address of host server
        :param port: port number of server
        :param host_pb_key: public key of server
        :param n_msgs: number of messages to send
        """
        # client instance bound to this client application + setup relay chain for this client + set host pb key
        self.client = Client(client_address)
        self.client.set_relays_chain(relays,n_layer)
        self.client.set_host_pb_key(host_pb_key)
        # relays chain through which the client sends messages
        self._relays = relays
        # server host address (ip address)
        self._host = host
        # server port
        self._port = port
        # rides history of client
        self._rides_history = pd.DataFrame(columns=COLS)

        # set up threads
        self._thread_app = threading.Thread(target=self.demo_client, args=(n_msgs,), name=str(self))
        # todo: this could be used as a real client user (not pc) with user input
        # self._thread_app = threading.Thread(target=self.send_message, args=*args)

    def __str__(self) -> str:
        return f'ClientApp-{self.client.address}'

    def __repr__(self) -> str:
        return 'ClientApp'

    def start_app(self) -> None:
        """
        start the thread bound to this client app
        :return:
        """
        self._thread_app.start()

    def join_app(self) -> None:
        """
        join the thread bound to this client app
        :return:
        """
        self._thread_app.join()

    def send_message(self, line: int, operator: str, code: int, boarding: str, st_src: str, st_dst: str) -> None:
        """
        send a message to the MoT servers
        :param line: line number of bus
        :param operator: operator of public transportation company
        :param code: travel code of ride
        :param boarding: boarding time of ride
        :param st_src: source station of ride
        :param st_dst: destination station of ride
        :return:
        """
        mot_msg = MotMessage(line, operator, code, boarding, st_src, st_dst)
        core_msg = POST + mot_msg.get_formatted_message() + END
        self.client.send_through_chain(self._host, self._port, core_msg)

    def send_gudance(self,msg_idx):
        '''
        send update parameter for server
        duo to the size limination
        we send some key parameter to enc
        :para message index
        :para bithch_size
        :para learning rate
        '''

        bitch_size=random.randint(5000,8000)
        lr=random.uniform(0,1)

        # with open("../checkpoints/cifar_cnn.pkl",'rb') as r:
        #     guidance=pickle.load(r)

        msg=str(msg_idx).encode() + b';' + str(bitch_size).encode()+ b';' + str(lr).encode()
        core_msg = POST + msg + END
        self.client.send_through_chain(self._host,self._port, core_msg)

    def get_rides_history(self) -> pd.DataFrame:
        """
        :return: get rides history of client
        """
        return self._rides_history

    def demo_client(self, n_msgs: int) -> None:
        """
        demonstrate an action of client-app
        :param n_msgs: number of messages to send to MoT
        :return:
        """
        # call to sleep, so the os scheduler queue the client thread to run after all the relays are setup
        time.sleep(1)
        gride = ride_generator(n_msgs)
        for i in range(n_msgs):
            print(f'{self.client}: sending message...\n')

            self.send_gudance(i)

            # create delay between sent messages
            time.sleep(1)
        print(f'{self.client} done.\n')
