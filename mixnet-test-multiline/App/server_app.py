from collections import deque
import threading
import pandas as pd

from NetworkNode import *
from App.message_app import COLS

# delimiter of different data fields inside a sent MotMessage
DELIM = ';'


class ServerApp:
    """
    represents the server application side
    """

    def __init__(self, host: str, port: int, name: str = 'ServerApp', n_mesg:int=10) -> None:
        """
        init a server application instance
        :param host: ip address of server
        :param port: port number of server
        :param name: name of application (optional)
        """
        # name of server application
        self.name = name
        # network server instance
        self.server = Server(host, port)
        self.server._total_mesg=n_mesg
        # buffer to pass to the server receive method to store received messages
        self._buffer = deque()
        # init app and server threads
        self._thread_server = threading.Thread(target=self.server.receive, args=(self._buffer,), name=str(self.server))
        self._thread_app = threading.Thread(target=self.receive_messages, name=str(self))
        # rides database dataframe of server application
        self._rides_database = pd.DataFrame(columns=COLS)

    def __str__(self) -> str:
        return f'{self.name}-{self.server.address}'

    def __repr__(self) -> str:
        return self.name

    def start_app(self) -> None:
        """
        start threads of server-app and network server
        :return:
        """
        self._thread_server.start()
        self._thread_app.start()

    def join_app(self) -> None:
        """
        join threads of server-app and network server
        :return:
        """
        self._thread_server.join()
        self._thread_app.join()

    def receive_messages(self) -> None:
        """
        receive message from the server and add it to the server data base
        :return:
        """
        n = 0
        print(f'{self}: ready to receive...\n')
        while True:
            if len(self._buffer) > 0:
                msg = self._buffer.popleft()
                # self._add_ride_to_database(msg)
                n += 1
                # msg_parsed = self._parse_msg(msg)
                print(f'{self} received: {msg}\n')
            if not self.server.is_connected():

                self.close_app()
                break
        print(f'{self} disconnecting...\n')

    def close_app(self) -> None:
        """
        close the socket of the server
        :return:
        """
        self.server.close_socket()

    def _add_ride_to_database(self, msg: bytes) -> None:
        """
        parse a ride message and add it to the server database
        :param msg: message received from the network
        :return:
        """
        self._rides_database.loc[len(self._rides_database)] = msg.decode('utf-8').split(DELIM)
