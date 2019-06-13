import numpy as np


def L_test():
    #
    raise NotImplementedError


relm_test_dict = {'L_test': L_test}
'''
RELM workflow

1. Load bin file to gdf
2. Generate bin rates:
    a. sort sources into spatial bins
    b. make spatial/magnitude bins
    c. add rates per magnitude per source per bin
3. Sort observed earthquakes into spatial/magnitude bins
4. Make stochastic event sets
    a. from sources? from bins?
    b. make one long one, then break up into many windows (overlapping? not?)
5. run tests


'''
