def extract_digits(s:str):
    return int(''.join(c for c in s if c.isdigit()))

def bin_to_char(digit):
    map = {0: '.', 1: '@', 2: 'S', 3:'G'}
    return map[digit]

def manhattan_dist(loc1, loc2):
    return abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1])
