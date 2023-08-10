# builtin modules
from __future__ import annotations
import socket
from collections import namedtuple
import random
from typing import List, Tuple

# project modules
from NetworkNode.server import Server
from NetworkNode.node import Node, MSG_MAX_SIZE, POST, DEST, PORT, DEBUG_MODE, SYM_KEY_LEN
from NetworkNode.utils import *

# pool size limit of each mixnode/relay
POOL_SIZE = 5
# represents a packet inside the mixnet
Packet = namedtuple('Packet', ['msg', 'dest', 'port'])


class Relay(Server):
    """
    represents a relay/MixNode insdie the mixnet
    """

    def __init__(self, address: str, port: int, keys: Tuple[str, str] = ('relay_pr_key', 'relay_pb_key')) -> None:
        """
        init a relay/mixnode
        :param address: ip address of the relay/mixnode
        :param port: port number of the relay
        :param keys: private and public keys of the realy
        """
        super().__init__(address, port, keys)
        # next relay in the chain
        self.next = None
        # previous relay in the chain
        self.prev = None
        # messages pool
        self._msgpool = set()

    def __str__(self) -> str:
        return f'Relay-{self.address}'

    def __repr__(self) -> str:
        return 'Relay'

    def __eq__(self, other: Relay) -> bool:
        return self.address == other.get_ip_address() \
               and self.port == other.port \
               and self.get_public_key() == other.get_public_key()

    def __ne__(self, other: Relay) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.address, self.port, self.get_public_key()))

    @staticmethod
    def setup_relay_chain(relays: List[Relay]) -> None:
        """
        sets-up the linked relay chain list according to how they are ordered in the given list:
        relay at index 0 is set to be the head, and relay at index -1 is set to be the tail

        :param relays: relays to chain up
        :return:
        """
        if len(relays) < 2:
            return
        relays[0].next = relays[1]  # setup next of head relay
        relays[-1].prev = relays[-2]  # setup previous of tail relay
        # setup next and previous for all other relays
        for i in range(1, len(relays) - 1):
            relays[i].next = relays[i + 1]
            relays[i].prev = relays[i - 1]

    @staticmethod
    def print_relay_chain(relays: List[Relay]) -> None:
        """
        pretty print the relays chain
        :param relays:
        :return:
        """
        for relay in relays:
            print(f'{relay.prev}<-{relay}->{relay.next}')

    def receive(self, **kwargs) -> None:
        """
        receive a message for a client, or possibly previous relay inside the chain
        :param kwargs:
        :return:
        """
        while True:
            # try to receive data from the socket until bytes are received or until timeout
            try:
                sock_conn, addr = self._socket.accept()
            # if reached timeout close the socket
            except (OSError, socket.timeout):
                self.close_socket()
                break
            # print(f"{self.address}: Connected by {addr}")
            data = sock_conn.recv(MSG_MAX_SIZE)
            # assert len(data) == MSG_MAX_SIZE, f'size is {len(data)}'
            msg_plain = self._decrypt_layer(data)
            # print(f'{self}: got message: {msg_plain}')
            # print(f'{self}: got message.')
            sock_conn.close()
            # parse message and send to destination
            packet = self._parse_msg(msg_plain)
            self._msgpool.add(packet)
            self._send_batch()
            # if time.time() - self._spawn >= self._ttl:
            #     pass

    def send(self, host: str, port: int, msg: bytes) -> None:
        """
        send a message to host::port
        :param host: address of host
        :param port: port number of host
        :param msg: message to be sent
        :return:
        """
        # print(f'{self}: sending...', end='')
        super().send(host, port, msg)

    def _parse_msg(self, msg: bytes) -> Packet:
        """
        parse the given message, extract the next layer, ip address and port number of next hop,
        and store it inside a packet instance
        :param msg: message the peel
        :return: packet format of the message
        """
        msg = Node.unwrap_message(msg)
        # msg format: <random bytes>;<next-layer>;<dest-address>;<port>
        start_idx = msg.find(POST)
        if start_idx == -1:
            raise ValueError('could not parse given message: it does not follows the correct format')
        dest_idx = msg.rfind(DEST)
        next_layer = msg[start_idx + len(POST):dest_idx]
        port_idx = msg.rfind(PORT)
        dest = msg[dest_idx + len(DEST):port_idx]
        # end_idx = msg.rfind(END)
        port = msg[port_idx + len(PORT):]
        return Packet(next_layer, dest, int(port))

    def _send_batch(self) -> None:
        """
        start sending POOL_SIZE messages
        :return:
        """
        # check if message pool reached the pool limit
        if len(self._msgpool) >= POOL_SIZE:
            # get the next batch and update the message pool of relay
            limit = min(POOL_SIZE, len(self._msgpool))
            # sample limit packets from message pool and send them
            batch = random.sample(self._msgpool, limit)
            # update message pool: remove sent packets
            self._msgpool.difference_update(batch)
            # shuffle the messages
            batch = list(batch)
            random.shuffle(batch)
            # send packets in chosen batch
            for packet in batch:
                # add random bytes to message: all sent messages in the mixnet should have the same size
                wrapped_msg = Node.wrap_message(packet.msg)
                self.send(packet.dest, packet.port, wrapped_msg)

    def _decrypt_layer(self, layer: bytes) -> bytes:
        """
        decrypt the given onion layer according to the onion routing protocol
        :param layer: layer to decrypt
        :return: decrypted layer
        """
        # if in debug mode just cut the symmetric key part and discard it
        if DEBUG_MODE:
            return layer
        # otherwise, extract the encrypted symmetric key part, and the cipher message part and:
        #   1. decrypt the symmetric key the relay's private key
        #   2. decrypt the cipher message with the symmetric key
        else:
            enc_key = layer[:SYM_KEY_LEN]
            enc_layer = layer[SYM_KEY_LEN:]
            plain_key = decrypt(self._pr_key, enc_key)
            return decrypt_symm(plain_key, enc_layer)
