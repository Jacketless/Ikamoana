from parcels import *
from Behaviour import *
import numpy as np
from SEAPODYM_functions import *
from argparse import ArgumentParser


def FADRelease(grid, location=[0, 0], individuals=100, start=None, timestep=21600, time=30,
               output_file='FADRelease', mode='scipy'):

    ParticleClass = JITParticle if mode == 'jit' else Particle

    fadset = grid.ParticleSet(size=individuals, pclass=ParticleClass, start=[location[0]-0.5, location[1]-0.5], finish=[location[0]+0.5, location[1]+0.5])

    starttime = grid.time[0] if start is None else start

    print("Starting Sim")
    fadset.execute(AdvectionRK4,
                    starttime=starttime, endtime=starttime+time*timestep, dt=timestep,
                    output_file=fadset.ParticleFile(name=output_file+"_results"),
                    interval=timestep)#, density_field=density_field)

if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int,  default=None,
                   help='Generate grid file with given dimensions')
    p.add_argument('-f', '--files', default=['/short/e14/jsp561/OFAM/ocean_u_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_v_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_phy_1993_01.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-v', '--variables', default=['U', 'V', 'H'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u', 'v', 'phy'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['lat', 'lon', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-m', '--map_dimensions', default=['yu_ocean', 'xu_ocean', 'Time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-t', '--time', type=int, default=15,
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-o', '--output', default='TagRelease',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=86400,
                   help='Length of timestep in seconds, defaults to one day')
    p.add_argument('-s', '--starttime', type=int, default=None,
                   help='Time of tag release (seconds since 1-1-1970)')
    p.add_argument('-l', '--location', type=float, nargs=2, default=[180,0],
                   help='Release location (lon,lat)')
    args = p.parse_args()
    filename = '/short/e14/jsp561/' + args.output

    U = args.files[0]
    V = args.files[1]
    H = args.files[2]

    filenames = {'U': args.files[0], 'V': args.files[1]}
    variables = {'U': args.netcdf_vars[0], 'V': args.netcdf_vars[1]}
    dimensions = {'lon': args.map_dimensions[1], 'lat': args.map_dimensions[0], 'time': args.map_dimensions[2]}

    grid = Grid.from_netcdf(filenames=filenames, variables=variables, dimensions=dimensions, vmin=-200, vmax=200)

    FADRelease(grid, location=args.location, start=args.starttime, individuals=args.particles,
               timestep=args.timestep, time=args.time, output_file=filename, mode=args.mode)
