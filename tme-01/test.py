import sys
def test():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print(sys.argv[1])
    return

test()