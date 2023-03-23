import re

def extract_number(filename):
    num = re.findall(r'\d+', filename)
    return int(num[0]) if num else 0
