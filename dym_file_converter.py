from parcels import Field, Grid
from SEAPODYM_functions import *
from argparse import ArgumentParser
import numpy as np
import math


def dym2netcdf(dymfile, outputfile, variable="SEAPODYM_Var"):
    if variable is None:
        variable="SEAPODYM_Var"
    dymfield = Field_from_DYM(dymfile, variable)

    dymfield.write(outputfile)


def dymKconversion(dymfile, outputfile):
    dymfield = Field_from_DYM(dymfile, 'K')
    scale_m = np.zeros(np.shape(dymfield.data[0, :, :]), dtype=np.float32)
    seconds_in_month = 30*24*60*60
    for i in range(np.shape(scale_m)[1]):
        for j in range(np.shape(scale_m)[0]):
            scale_m[j, i] = ((1000*1.852*60) * (1000*1.852*60*math.cos(dymfield.lat[j]*math.pi/180)))/seconds_in_month
    print(scale_m)
    for t in range(len(dymfield.time)):
        dymfield.data[t, :, :] *= scale_m
    dymfield.write(outputfile)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('-d', '--dymfile', type=str,
                   help='file name of .dym file')
    p.add_argument('-o', '--outputfile', type=str,
                   help='output filename')
    p.add_argument('-v', '--variable', type=str, default=None,
                   help='name of variable in .dym file')
    p.add_argument('-s', '--scale', type=bool, default=False,
                   help='scale dym file from degrees/month')

    args = p.parse_args()
    if args.scale is True:
        dymKconversion(args.dymfile, args.outputfile)
    else:
        dym2netcdf(args.dymfile, args.outputfile, args.variable)
