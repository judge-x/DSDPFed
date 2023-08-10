import sys
import time
import socket
from secrets import token_bytes
from typing import Tuple, Any

from NetworkNode.utils import *

SOCKET_TIMEOUT = 60
MSG_MAX_SIZE = 8192
CORE_MSG_SIZE = 256
PSEUDONYM_LEN = 8
TTL = 1000  # time to live
SYM_KEY_LEN = 256
SLEEP_SEC = 1
MAX_TRIES = 10

POST = b'POST'
DEST = b'DEST'
PORT = b'PORT'
END = b'END'

# MSG_FORMAT = f'{{r}}{POST}{{m}}{DEST}{{d}}{PORT}{{p}}'
# MSG_FORMAT = '{r}POST{m}DEST{d}PORT{p}'
UTF8 = 'utf-8'
ISO8859 = 'ISO 8859-1'

DEBUG_MODE = False


class Node:
    """
    represents a general node inside the network
    """

    def __init__(self, address: str, keys: Tuple[str, str] = ('node_pr_key', 'node_pb_key')) -> None:
        """
        init a network node
        :param address: ip address of node
        :param keys: private and public keys of the node
        """
        self.address = address
        # init keys
        pr_key, pb_key = load_key_pair(keys)
        # private key of client
        self._pr_key = pr_key
        # public key of client
        self._pb_key = pb_key
        # todo: ttl can be used instead of socket timeout
        # set time to live and time created for later disconnecting the node
        self._ttl = TTL
        self._spawn = time.time()

    def get_public_key(self) -> rsa.RSAPublicKey:
        """
        :return: public key of node
        """
        return self._pb_key

    def get_ip_address(self) -> str:
        """
        :return: ip address of node
        """
        return self.address

    def receive(self, *args) -> Any:
        """
        abstract method to be implemented (optionally) be the child class
        :param args: optional arguments
        :return:
        """
        pass

    @staticmethod
    def send(host: str, port: int, msg: bytes) -> None:
        """
        send given message to host::port
        :param host: ip address of host
        :param port: port number of host
        :param msg: message to be sent
        :return:
        """
        # open TCP connection socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for i in range(MAX_TRIES):
            try:
                # connect to host::port

                try:
                    s.connect((host, port))
                    print("连接成功")
                except Exception as e:
                    print("连接失败:", str(e))
                # send message, make sure all bytes was sent successfully
                s.sendall(msg)
                # close socket
                s.close()
                return
            except (OSError, TimeoutError, ConnectionError):
                continue
        print('failed to send message', file=sys.stderr)

    @staticmethod
    def format_message(msg: bytes, dest: bytes, port: bytes) -> bytes:
        """
        format the given message according to the onion routing protocol
        :param msg: data/payload
        :param dest: ip address of destination
        :param port: port number of destination
        :return: formatted message
        """
        return token_bytes(PSEUDONYM_LEN) + POST + msg + DEST + dest + PORT + port + END

    @staticmethod
    def wrap_message(msg: bytes) -> bytes:
        """
        wrap the given message with random bytes. this in order to make all the messages that are being
        sent to, inside, and outside the mixnet to be the same size
        :param msg: message to wrap with random bytes
        :return: wrapped message
        """
        return msg + token_bytes(MSG_MAX_SIZE - len(msg))

    @staticmethod
    def unwrap_message(msg: bytes) -> bytes:
        """
        unwrap the given message from the added random bytes
        :param msg: message wrapped with random bytes (decrypted)
        :return: unwrapped message
        """
        # find start index of end message symbol
        end_idx = msg.rfind(END)
        if end_idx == -1:
            raise ValueError('could not unwrap message')
        # extract the message content
        return msg[:end_idx]
