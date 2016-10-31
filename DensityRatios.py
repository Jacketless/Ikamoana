from netCDF4 import Dataset
import numpy as np
from argparse import ArgumentParser
from parcels import Field


def calculateDensityRatio(dfiles, output):
    Fields = []
    for dfile in dfiles:
        print("Loading %s" % dfile)
        Fields.append(Field.from_netcdf(dfile,
                                        dimensions={'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'TagDensity'},
                                        filenames=[dfile]))

    limits = [0, 0, 0, 0]
    limits[0] = np.max([field.lon[0] for field in Fields])
    limits[1] = np.min([field.lon[-1] for field in Fields])
    limits[2] = np.max([field.lat[0] for field in Fields])
    limits[3] = np.min([field.lat[-1] for field in Fields])
    #limits[1] = (np.min(field.lon[-1]) for field in Fields)
    #limits[2] = (np.max(field.lat[0]) for field in Fields)
    #limits[3] = (np.min(field.lat[-1]) for field in Fields)
    time_lim = [np.max([field.time[0] for field in Fields]), np.min([field.time[-1] for field in Fields])]
    #time_lim = [Fields[0].time[0], Fields[0].time[-1]]

    lon = np.arange(start=limits[0], stop=limits[1]+1, dtype=np.float32)
    lat = np.arange(start=limits[2], stop=limits[3]+1, dtype=np.float32)
    time = np.arange(time_lim[0], time_lim[1]+1, 30*24*60*60, dtype=np.float32)
    Ratio = np.zeros([len(time), len(lat), len(lon)], dtype=np.float32)
    print(limits)
    print(Fields[0].lon)
    print(Fields[1].lon)
    print(lon)
    print(Fields[0].lat)
    print(Fields[1].lat)
    print(lat)
    print(Fields[0].time)
    print(Fields[1].time)
    print(time)
    for t in range(len(time)):
        for x in range(len(lon)):
            for y in range(len(lat)):
                tagged = Fields[0].data[np.where(Fields[0].time == time[t])[0][0],
                                     np.where(Fields[0].lat == lat[y])[0][0],
                                     np.where(Fields[0].lon == lon[x])[0][0]]
                pop = Fields[1].data[np.where(Fields[1].time == time[t])[0][0],
                                     np.where(Fields[1].lat == lat[y])[0][0],
                                     np.where(Fields[1].lon == lon[x])[0][0]]
                #print('%s - %s' % (tagged, pop))
                if pop == 0:
                    Ratio[t, y, x] = 0
                else:
                    Ratio[t, y, x] = tagged/pop

    Ratios = Field('DensityRatio', Ratio, lon, lat, time=time)
    Ratios.write(filename=output)


def Ratio_Test(dfile):
    Field.from_netcdf(dfile, dimensions={'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'TagDensity'},
                      filenames=[dfile])


if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('-d', '--dfiles', nargs=2, default='none',
                   help='Particle density files')
    p.add_argument('-o', '--output', default='DensityRatio',
                   help='Output file name')

    args = p.parse_args()

    if args.dfiles == 'none':
        print("No particle file provided!")
    else:
        calculateDensityRatio(args.dfiles, output=args.output)
