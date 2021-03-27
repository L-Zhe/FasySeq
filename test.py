import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--A', type=int, default=10)
args = parser.parse_args()
def add(args):
    setattr(args, 'B', 10)

add(args)
print(args.B)
