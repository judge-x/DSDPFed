from cryptography.exceptions import InvalidSignature
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
import os
import pickle
import json

KEYS_PATH = os.path.abspath('keys')
JSON_PATH = os.path.abspath('json')


# ============================================ KEYS_FUNCTIONS ============================================ #
def generate_key_pair(pr_name: str, pb_name: str) -> (rsa.RSAPrivateKey, rsa.RSAPublicKey):
    pr_key_file = f'{KEYS_PATH}\\{pr_name}.pem'
    pb_key_file = f'{KEYS_PATH}\\{pb_name}.pem'

    # generate key pair
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key = private_key.public_key()

    # serialize keys
    pem_pr = private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                       format=serialization.PrivateFormat.PKCS8,
                                       encryption_algorithm=serialization.NoEncryption())
    pem_pb = public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                     format=serialization.PublicFormat.SubjectPublicKeyInfo)
    # write keys
    with open(pr_key_file, 'wb') as file:
        file.write(pem_pr)
    with open(pb_key_file, 'wb') as file:
        file.write(pem_pb)
    return private_key, public_key


def load_key_pair(key_pair: [tuple, list]) -> (rsa.RSAPrivateKey, rsa.RSAPublicKey):
    assert isinstance(key_pair, tuple) or isinstance(key_pair, list)
    pr_name, pb_name = key_pair
    pr_key_file = f'{KEYS_PATH}/{pr_name}.pem'
    pb_key_file = f'{KEYS_PATH}/{pb_name}.pem'
    if os.path.exists(pr_key_file) and os.path.exists(pb_key_file):
        with open(pr_key_file, 'rb') as file:
            private_key = serialization.load_pem_private_key(file.read(), password=None, backend=default_backend())
        with open(pb_key_file, 'rb') as file:
            public_key = serialization.load_pem_public_key(file.read(), backend=default_backend())
        return private_key, public_key
    else:
        return generate_key_pair(pr_name, pb_name)


def generate_key(key_name: [str, None] = None) -> bytes:
    if key_name is None:
        key_filename = f'{KEYS_PATH}/sym_key.key'
    else:
        key_filename = f'{KEYS_PATH}/{key_name}.key'
    if os.path.exists(key_filename):
        return load_key(key_filename)
    key = Fernet.generate_key()
    with open(key_filename, 'wb') as output_key:
        output_key.write(key)
    return key


def load_key(key_name: str) -> bytes:
    if not os.path.exists(f'{KEYS_PATH}/{key_name}.key'):
        return generate_key(key_name)
    else:
        with open(f'{KEYS_PATH}/{key_name}.key', 'rb') as key_file:
            return key_file.read()


def get_key_bytes_format(key) -> bytes:
    if isinstance(key, rsa.RSAPrivateKey):
        return key.private_bytes(encoding=serialization.Encoding.DER,
                                 format=serialization.PrivateFormat.PKCS8,
                                 encryption_algorithm=serialization.NoEncryption())
    elif isinstance(key, rsa.RSAPublicKey):
        return key.public_bytes(encoding=serialization.Encoding.DER,
                                format=serialization.PublicFormat.SubjectPublicKeyInfo)
    else:
        raise TypeError('given key must be of type rsa')


# ============================================ ASYMMETRIC_ENCRYPTION ============================================ #
def encrypt(key: rsa.RSAPublicKey, message: bytes) -> bytes:
    assert isinstance(message, bytes)
    pad = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    cipher_text = key.encrypt(message, pad)
    return cipher_text


def decrypt(key: rsa.RSAPrivateKey, message: bytes) -> [str, bytes]:
    pad = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    plain_text = key.decrypt(message, pad)
    return plain_text


def sign(pr_key: rsa.RSAPrivateKey, message: bytes) -> bytes:
    signature = pr_key.sign(message,
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                            hashes.SHA256())
    return signature


def verify(pb_key: rsa.RSAPublicKey, message: bytes, signature: bytes) -> bool:
    try:
        pb_key.verify(signature,
                      message,
                      padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                      hashes.SHA256())
    except InvalidSignature:
        return False
    return True


# ============================================ SYMMETRIC_ENCRYPTION ============================================ #

def encrypt_symm(key: bytes, message: [str, bytes]):
    if isinstance(message, str):
        byte_msg = message.encode()
    else:
        byte_msg = message
    fernet_key = Fernet(key)
    cipher_txt = fernet_key.encrypt(byte_msg)
    return cipher_txt


def decrypt_symm(key: bytes, message: bytes):
    fernet_key = Fernet(key)
    plain_txt = fernet_key.decrypt(message)
    return plain_txt


# ============================================ FILE_FUNCTIONS ============================================ #


def load_pickle(filename):
    with open(filename, 'rb') as pkl:
        return pickle.load(pkl)


def save_pickle(filename, data):
    with open(filename, 'wb') as pkl:
        pickle.dump(data, pkl)


def save_json(filename, data):
    json_str = json.dumps(data, indent=4)
    with open(f'{JSON_PATH}/{filename}', 'w') as file:
        file.write(json_str)


def load_json(filename):
    with open(f'{JSON_PATH}/{filename}', 'r') as file:
        return json.load(file)
