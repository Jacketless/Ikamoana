from SEAPODYM_PARCELS import *


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('-t', '--time', type=int, default=30,
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-o', '--output', default='SIMPODYM_test_2003',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=172800,
                   help='Length of timestep in seconds, defaults to two days')
    args = p.parse_args()

    dimensions = ['lat', 'lon', 'time']
    netcdf_vars = ['u', 'v', 'habitat']

    SIMPODYM(forcingU='SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc',
             forcingV='SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc',
             forcingH='SEAPODYM_Forcing_Data/SEAPODYM2003_HABITAT_Prepped.nc',
             startD='SEAPODYM_Forcing_Data/SEAPODYM2003_DENSITY_Prepped.nc',
             Uname=netcdf_vars[0], Vname=netcdf_vars[1], Hname=netcdf_vars[2],
             dimLat=dimensions[0], dimLon=dimensions[1], dimTime=dimensions[2],
             individuals=args.particles, timestep=args.timestep, time=args.time,
             mode=args.mode, output_file=args.output)
