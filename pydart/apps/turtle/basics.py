import numpy as np

def flatten(l):
	return list(_flatten_(l))

def _flatten_(*args):
    for x in args:
        if hasattr(x, '__iter__'):
            for y in _flatten_(*x):
                yield y
        else:
            yield x
def check_valid_data(data):
	d = flatten(data)
	return not np.isnan(d).any()