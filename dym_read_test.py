from SEAPODYM_functions import *
import numpy as np


if __name__ == "__main__":

    data = Field_from_DYM('sim_4Joe/2003-2007/skj_diffusion.dym', name='V', fromyear=2003, frommonth=1, toyear=2005, tomonth=5)

    print(data.data)
    data.write('dymtest')