# python imports
import socket
import time
from collections import deque
from typing import Tuple

# project imports
from NetworkNode.node import Node, SOCKET_TIMEOUT, POST, MSG_MAX_SIZE, DEBUG_MODE, CORE_MSG_SIZE, SLEEP_SEC
from NetworkNode.utils import *
# from eval import N_ALL_MSGS


class Server(Node):
    """
    represents a server in the network
    """

    def __init__(self, address: str, port: int, keys: Tuple[str, str] = ('server_pr_key', 'server_pb_key')) -> None:
        """
        init a server instance
        :param address: ip address of the server
        :param port: port number of the server
        :param keys: private and public keys filenames of the server
        """
        super().__init__(address, keys)
        self.port = port
        # setup socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((address, port))
        self._socket.settimeout(SOCKET_TIMEOUT)  # setup timeout for the socket
        self._socket.listen()  # setup as a listening socket
        self._socket_closed = False  # flag to indicate if the socket has been closed

        self._start_time=time.time()
        self._end_time=None

        self._total_mesg=1
        self._received_mesg=0

    def __str__(self) -> str:
        return f'Server-{self.address}'

    def __repr__(self) -> str:
        return 'Server'

    def get_port(self) -> int:
        """
        :return: port number of server
        """
        return self.port

    def receive(self, buffer: [deque, list]) -> None:
        """
        receive a message
        :param buffer: a buffer to pushed into the received message
        :return:
        """
        print(f'{self} listening...\n')
        while True:
            try:
                sock_conn, addr = self._socket.accept()
            except (OSError, socket.timeout):
                self._end_time=time.time()
                self.close_socket()
                print("Server end wiht timeout.....")
                print("the received rete is:" ,self._received_mesg/self._total_mesg)
                break
            # print(f"{self.address}: Connected by {addr}")
            data = sock_conn.recv(MSG_MAX_SIZE)
            # print(f'{self}: got data: {data}')
            msg_plain = self._decrypt_msg(data)
            msg_parsed = self._parse_msg(msg_plain)
            buffer.append(msg_parsed)
            self._received_mesg+=1
            print(f'{self}: got message: {msg_parsed}')

            sock_conn.close()
            if(self._received_mesg==int(self._total_mesg*0.8)):
                print("Server end wiht full message.....")
                self._end_time=time.time()

                self.close_socket()
                break

            # if time.time() - self._spawn >= self._ttl:
            #     self.close_socket()
            #     break

    def close_socket(self) -> None:
        """
        close the socket of the server
        :return:
        """
        if not self._socket_closed:
            self._socket_closed = True
            self._socket.close()
            print(f'{self}: disconnecting\n')
            time.sleep(SLEEP_SEC)

    def is_connected(self) -> bool:
        """
        :return: true if socket is open, false otherwise
        """
        return not self._socket_closed

    def _parse_msg(self, msg: bytes) -> bytes:
        """
        parses the given message. unwrap the given message: remove added random bytes
        :param msg: message to parse
        :return: unwrapped and parsed message
        """
        msg = Node.unwrap_message(msg)
        start_idx = msg.find(POST)
        if start_idx == -1:
            raise ValueError('could not parse given message: it does not follows the correct format')
        return msg[start_idx + len(POST):]

    def _decrypt_msg(self, msg: bytes) -> bytes:
        """
        decrypt the given message
        :param msg: message to decrypt
        :return: decrypted message
        """
        unwrapped_msg = msg[:CORE_MSG_SIZE]
        if DEBUG_MODE:
            return unwrapped_msg
        else:
            return decrypt(self._pr_key, unwrapped_msg)
