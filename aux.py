def extract_digits(s:str):
    return int(''.join(c for c in s if c.isdigit()))