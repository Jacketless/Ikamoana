from parcels import *
from Behaviour import *
import numpy as np
from SEAPODYM_functions import *
from SEAPODYM_PARCELS import *
from argparse import ArgumentParser


def TagRelease(grid, location=[0, 0], spread = 0, individuals=100, start=None, timestep=86400, time=30,
               output_file='TagRelease', mode='scipy', start_age=4):

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle
    SKJ = Create_TaggedFish_Class(ParticleClass)
    lats = []
    lons = []
    for _ in range(individuals):
        lats.append(location[1] + np.random.uniform(spread*-1, spread))
        lons.append(location[0] + np.random.uniform(spread*-1, spread))

    fishset = grid.ParticleSet(size=individuals, pclass=SKJ, lon=lons, lat=lats)

    age = fishset.Kernel(AgeIndividual)
    diffuse = fishset.Kernel(LagrangianDiffusion)
    advect = fishset.Kernel(Advection_C)
    taxis = fishset.Kernel(GradientRK4_C)
    move = fishset.Kernel(Move)
    checkrelease = fishset.Kernel(CheckRelease)

    starttime = 0 if start is None else start
    for p in fishset.particles:
        p.setAge(start_age)
        p.release_time = grid.time[0] + starttime
        #p.active.to_write = True

    grid.write(output_file)

    print("Starting Sim")
    fishset.execute(checkrelease + age + taxis + advect + diffuse + move,
                    starttime=grid.time[0], endtime=grid.time[0]+time*timestep, dt=timestep,
                    output_file=fishset.ParticleFile(name=output_file+"_results"),
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
    p.add_argument('-f', '--files', default=['SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc', 'SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc', 'SEAPODYM_Forcing_Data/SEAPODYM2003_HABITAT_Prepped.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-v', '--variables', default=['U', 'V', 'H'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u', 'v', 'habitat'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['lat', 'lon', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-m', '--map_dimensions', default=['lat', 'lon', 'time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-t', '--time', type=int, default=15,
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-o', '--output', default='TagRelease',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=86400,
                   help='Length of timestep in seconds, defaults to one day')
    p.add_argument('-l', '--location', type=float, nargs=2, default=[180,0],
                   help='Release location (lon,lat)')
    p.add_argument('-a', '--age', type=int, default=4,
                   help='Starting age of fish at beginning of sim')
    p.add_argument('-c', '--cluster', type=float, default = 0.1,
                   help='+- spatial range around release location (uniformly distributed, in degrees lat/lon)')
    p.add_argument('-s', '--start', type=float, default = 4,
                   help='age in months when fish are released')
    p.add_argument('-ds', '--diffusion_scale', type=float, default = 4,
                   help='scaler for diffusion')


    args = p.parse_args()
    filename = args.output

    U = args.files[0]
    V = args.files[1]
    H = args.files[2]

    grid = Create_SEAPODYM_Grid(forcingU=U, forcingV=V, forcingH=H,
             Uname=args.netcdf_vars[0], Vname=args.netcdf_vars[1], Hname=args.netcdf_vars[2],
             dimLat=args.dimensions[0], dimLon=args.dimensions[1], dimTime=args.dimensions[2],
                                start_age=11, diffusion_scale=args.diffusion_scale)
    print("Calculating H Gradient Fields")
    gradients = grid.H.gradient()
    for field in gradients:
        grid.add_field(field)
    print("Calculating K Gradient Fields")
    K_gradients = grid.K.gradient()
    for field in K_gradients:
        grid.add_field(field)

    # The time in seconds from the start grid time when fish will become active (SIMPLEDYM grids start at age 4 months)
    start_time = (args.start - 4) * 30 * 24 * 60 * 60

    TagRelease(grid, location=args.location, spread=args.cluster, individuals=args.particles,
               start_age=args.age, timestep=args.timestep, time=args.time, output_file=args.output, mode=args.mode,
               start=start_time)
