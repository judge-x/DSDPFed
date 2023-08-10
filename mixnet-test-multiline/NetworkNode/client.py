# python imports
import random
from typing import List, Tuple
from secrets import token_bytes

# project imports
from NetworkNode.node import Node, PSEUDONYM_LEN, DEBUG_MODE, CORE_MSG_SIZE, MAX_TRIES
from NetworkNode.relay import Relay
from NetworkNode.utils import *


class Client(Node):
    """
    represents a client in the network
    """

    def __init__(self, address: str, keys: Tuple[str, str] = ('client_pr_key', 'client_pb_key')) -> None:
        """
        init a client instance
        :param address: ip address of the client
        :param keys: private and public keys of the client
        """
        super().__init__(address, keys)
        # set of all known relay nodes
        self._relays = set()
        # head relay in the chain
        self._head_relay = None
        # public key of host
        self._host_pb_key = None
        # symmetric key for onion encryption
        self._key_sym = load_key('client_key_sym')

    def __str__(self) -> str:
        return f'Client-{self.address}'

    def __repr__(self) -> str:
        return 'Client'

    def __eq__(self, other: Relay) -> bool:
        return self.address == other.get_ip_address() and self.get_public_key() == other.get_public_key()

    def __ne__(self, other: Relay) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.address, self.get_public_key()))

    def send(self, host: str, port: int, msg: bytes) -> None:
        """
        send the given message to host::port
        :param host: host ip address
        :param port: port number of host
        :param msg: message to send
        :return:
        """
        print(f'{self}: sending to ',host ,':',port )
        super().send(host, port, msg)

    def send_through_chain(self, host: str, port: int, msg: bytes) -> None:
        """
        send the given message to host::port through the mixnet chain
        :param host: host/server ip address: last destination in the onion layers
        :param port: port number of the host/server
        :param msg: message to be sent to the server
        :return:
        """
        # if no relays are known to the client, send original message directly to the server
        if self._head_relay is None:
            # add random bytes to the message
            self.send(host, port, token_bytes(PSEUDONYM_LEN) + msg)
        # send onion message through the known relays chain
        else:
            # add random bytes to the core-message
            core_msg = token_bytes(PSEUDONYM_LEN) + msg
            onion = self.onion_msg(host, port, core_msg, self._head_relay)
            # assert len(onion) <= MSG_MAX_SIZE, f'size is {len(onion)}'
            # print(f'onion size is: {len(onion)}')
            wrapped_onion = Node.wrap_message(onion)
            self.send(self._head_relay.address, self._head_relay.port, wrapped_onion)

    def get_relays(self) -> List[Relay]:
        """
        :return: list of the relays/mixnodes
        """
        return list(self._relays)

    def set_relays_chain(self, relays: List[Relay],n_layer:int) -> None:
        """
        setups a relays/mixnodes chain for the client
        :param relays: relays making the mixnet
        :return:
        """
        if len(relays) == 0:
            # raise ValueError('given list is empty')
            return
        # save relays set on the client side
        n_each_layer=len(relays)/n_layer

        self._relays = set(relays)
        # go backwards the chain, and find the head relay
        head=int(len(relays)-n_each_layer)
        tail=len(relays)-1
        current_relay = relays[random.randint(head,tail)]
        for i in range(n_layer-1):
            head=int(head-n_each_layer)
            tail=int(tail-n_each_layer)
            current_relay.prev=relays[random.randint(head,tail)]
            current_relay=relays[random.randint(head,tail)]

        # while current_relay.prev is not None:
        #     current_relay = current_relay.prev
        self._head_relay = current_relay

    def set_host_pb_key(self, pb_key: rsa.RSAPublicKey) -> None:
        """
        save public key of host/server on the client's side
        :param pb_key: public key of the server (end destination of a message)
        :return:
        """
        if pb_key is not None:
            self._host_pb_key = pb_key

    def onion_msg(self, host: str, port: int, msg: bytes, relay: Relay) -> bytes:
        """
        create msg following the onion encryption protocol.

        :param host: ip address of the host server: last destination in the chain
        :param port: port number of the host server
        :param msg: core msg to send to the server
        :param relay: relay corresponding to the current layer
        :return: onion message
        """
        # if went through all the chain, or no chain was set for this client:
        # return the original message (no need to onion anything)
        if len(self._relays) == 0 or relay is None:
            return encrypt(self._host_pb_key, msg)
        # if reached the last relay in the chain, return the last inner layer
        if relay.next is None:
            if DEBUG_MODE or self._host_pb_key is None:
                inner_layer = msg
            else:
                # encrypt core msg with host public key
                inner_layer = encrypt(self._host_pb_key, msg)
            # wrap inner layer with outer layer
            cur_layer = Node.format_message(inner_layer,
                                            host.encode(),
                                            str(port).encode())
            return self._encrypt_layer(relay.get_public_key(), cur_layer)
        # recursive call with the next relay in the chain
        else:
            cur_layer = Node.format_message(self.onion_msg(host, port, msg, relay.next),
                                            relay.next.get_ip_address().encode(),
                                            str(relay.next.get_port()).encode())
            return self._encrypt_layer(relay.get_public_key(), cur_layer)

    def _encrypt_layer(self, pb_key: rsa.RSAPublicKey, layer: bytes) -> bytes:
        """
        encrypt the given layer according to the onion routing protocol
        :param pb_key: public key of the network component (relay, or server) to peel the created layer
        :param layer: message to wrap
        :return: encrypted layer
        """
        # in debug mode, just concatenate with the plain message
        if DEBUG_MODE or pb_key is None:
            return layer
        # encrypt the client's symmetric key with the given public key and concatenate with the
        # message, encrypted with this symmetric key.
        else:
            return encrypt(pb_key, self._key_sym) + encrypt_symm(self._key_sym, layer)
