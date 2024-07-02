# Anonymizing Public Transportation Rides Information

## Usage

run `$ python3 main.py` from root directory of the project, with optional flags.

### Flags

`-h, --help`<br />          
show this help message and exit

`-d, --demo-mode`<br />   
run demonstration mode. clients send random messages to the server. this option executes `app_demo` in
`mot_app.py`

`-s, --server`<br />
setup a server to run on the machine. use `-a` to choose address for server. use `-p` to choose port for server.
otherwise, default ip address and port will be set.

`-c n_clients n_messages, --clients n_clients n_messages`<br />
setup `n_clients` to send `n_messages` to the server through the MixNet. use `-a` to choose address for server. use `-p`
to choose port for server. otherwise, default ip address and port will be set. maximal number of `n_clients`
,`n_messages` are 20,000, 128 respectively.

`-p server_port, --port server_port`<br />
port number of the MoT server

`-a server_address, --address server_address`<br />
ip address of the MoT server
