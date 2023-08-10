import random
import pandas as pd
import os

STATION_SOURCE = 'stationSrc'
STATION_DEST = 'stationDst'
BOARDING_TIME = 'boardingTime'
TRAVEL_CODE = 'travelCode'
OPERATOR = 'operator'
LINE_NUMBER = 'lineNumber'

OPERATORS = ['EGGED', 'TNUFA', 'SUPERBUS', 'KAVIM', 'AFIKIM', 'DAN', 'METROPOLIN']
LINE_NUMBERS = range(1, 1000)
TRAVEL_CODES = range(1, 10)
HOURS = [f'0{n}' if n < 10 else f'{n}' for n in range(24)]
MINUTES = [f'0{n}' if n < 10 else f'{n}' for n in range(0, 60, 5)]
COLS = [LINE_NUMBER, OPERATOR, TRAVEL_CODE, BOARDING_TIME, STATION_SOURCE, STATION_DEST]
ROWS = 50000

CITIES_FILENAME = 'cities.txt'
RIDES_EXAMPLE_FILE = 'rides_example.csv'

SEP = b';'
MOT_MSG_FORMAT = b'{ln};{op};{code};{brd};{st_src};{st_dst}'


class MotMessage:
    """
    MoT message format class.
    this class represents a message that follows the data needed for the MoT (ministry of transportation)
    """

    def __init__(self, line_number: int, operator: str, travel_code: int,
                 boarding_time: str, st_source: str, st_dest: str) -> None:
        """
        init a MoT message instance
        :param line_number: line number of bus
        :param operator: operator of public transportation company
        :param travel_code: travel code of ride
        :param boarding_time: boarding time of ride
        :param st_source: source station
        :param st_dest: destination station
        """
        self.line_number = line_number
        self.operator = operator
        self.travel_code = travel_code
        self.boarding_time = boarding_time
        self.st_source = st_source
        self.st_dest = st_dest

    def get_formatted_message(self) -> bytes:
        """
        :return: a formatted bytes message that follows the MoT API
        """
        return str(self.line_number).encode() + SEP \
               + self.operator.encode() + SEP \
               + str(self.travel_code).encode() + SEP \
               + self.boarding_time.encode() + SEP \
               + self.st_source.encode() + SEP \
               + self.st_dest.encode()


def gen_line_number(n: int):
    """
    line number generator helper for generate_rides_example_file
    :param n: amount to generate
    :return: line number generator
    """
    for i in range(n):
        yield random.choice(LINE_NUMBERS)


def gen_travel_code(n: int):
    """
    travel code generator helper for generate_rides_example_file
    :param n: amount to generate
    :return: travel code generator
    """
    for i in range(n):
        yield random.choice(TRAVEL_CODES)


def gen_operator(n: int):
    """
    operator name generator helper for generate_rides_example_file
    :param n: amount to generate
    :return: operator generator
    """
    for i in range(n):
        yield random.choice(OPERATORS)


def gen_boarding_time(n: int):
    """
    boarding time generator helper for generate_rides_example_file
    :param n: amount to generate
    :return: boarding time generator
    """
    for i in range(n):
        h = random.choice(HOURS)
        m = random.choice(MINUTES)
        yield f'{h}:{m}'


def gen_station(n: int, path: str):
    """
    station name generator helper for generate_rides_example_file
    :param n: amount to generate
    :param path: path name to station file
    :return: station generator
    """
    with open(path, 'r') as file:
        cities = file.readlines()
    for i in range(n):
        yield random.choice(cities).strip('\n')


def generate_rides_example_file() -> None:
    """
    generate public transportation rides example
    :return:
    """
    path = os.path.abspath(f'./App/{CITIES_FILENAME}')
    df = pd.DataFrame(index=range(ROWS), columns=COLS)
    lines = gen_line_number(ROWS)
    ops = gen_operator(ROWS)
    codes = gen_travel_code(ROWS)
    times = gen_boarding_time(ROWS)
    st_src = gen_station(ROWS, path)
    st_dst = gen_station(ROWS, path)
    for i in range(ROWS):
        df.loc[i] = [next(lines), next(ops), next(codes), next(times), next(st_src), next(st_dst)]
        if i % 100 == 0:
            print(f'{(i + 1) / ROWS * 100}%...')
    df.to_csv(f'./App/{RIDES_EXAMPLE_FILE}', index=False)


def ride_generator(n: int):
    """
    :param n: amount to generate
    :return: pd.DataFrame rides generator
    """
    df = pd.read_csv(f'./App/{RIDES_EXAMPLE_FILE}')
    for i in range(n):
        yield df.sample()
