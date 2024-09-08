import aux
import numpy as np


class instance:
    def __init__(self, map_f_name, verbose = True):
        self.map_f_name = map_f_name
        self.verbose = verbose
    def load_map(self):
        with open(self.map_f_name) as map_file:
            headers = [map_file.readline() for i in range(4)]
            self.num_of_rows = aux.extract_digits(headers[1])
            self.num_of_cols = aux.extract_digits(headers[2])
            file_string = map_file.read()
        self.map_size = self.num_of_rows * self.num_of_cols
        file_string = [c for c in file_string if c != '\n']
        self.map = np.array(list(map(lambda c: int(c != '.'), file_string))).reshape(self.num_of_rows, self.num_of_cols)
        if self.verbose:
            print(f'**** Successfully loaded map from: {self.map_f_name} ****')
            print(f'     number of rows: {self.num_of_rows}')
            print(f'     number of columns: {self.num_of_cols}')
    def print_map(self):
        for line in self.map:
            print(''.join(aux.bin_to_char(num) for num in line))
