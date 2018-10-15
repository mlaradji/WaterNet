"""Useful miscellaneous functions."""

def print_line():
    '''
    Print a solid line.
    '''
    print('____________________________________________________________________________________________________')
    print()
    

def counter():
    '''
    A simple counter generator object.
    
    Usage:
        A = counter()
        current_count = next(A)
    '''
    
    counter = 0
    while True:
        counter += 1
        yield counter