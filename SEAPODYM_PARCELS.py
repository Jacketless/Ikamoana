from parcels import *
import numpy as np
import math
from argparse import ArgumentParser

def SIMPODYM(forcingU, forcingV, forcingH, startD=None,
             Uname='u', Vname='v', Hname='habitat', Dname='density',
             dimLon='lon', dimLat='lat',dimTime='time',
             individuals=100, timestep=172800, time=30, start_age=4,
             output_density=FALSE, output_file="SIMPODYM"):

    filenames = {'U': forcingU, 'V': forcingV, 'H', forcingH, 'SEAPODYM_Density': startD}
    variables = {'U': Uname, 'V': Vname, 'H': Hname, 'SEAPODYM_Density': Dname}
    dimensions = {'lon': dimLon, 'lat': dimLat, 'time': dimTime}

    grid = Grid.from_netcdf(filenames=filenames, variables=variables, dimensions=dimensions, vmax=200)

    if output_density:
        # Add a density field that will hold particle densities
        grid.add_field(Field('Density', np.full([grid.U.lon.size, grid.U.lat.size, grid.U.time.size],-1, dtype=np.float64),
                       grid.U.lon, grid.U.lat, depth=grid.U.depth, time=grid.U.time, transpose=True))
        density_field = grid.Density

    # Offline calculate the 'diffusion' grid as a function of habitat
    K = Create_SEAPODYM_Diffusion_Field(grid.H, 24*60*60)
    grid.add_field(K)

    # Offline calculation of the diffusion and basic habitat grid
    gradients = grid.H.gradient()
    for field in gradients:
        grid.add_field(field)
    K_gradients = grid.K.gradient()
    for field in K_gradients:
        grid.add_field(field)

    ParticleClass = JITParticle if args.mode == 'jit' else Particle

    if startD is not None:
        grid.add_field(Field.from_netcdf('SEAPODYM_Density', dimensions=dimensions, filenames=startD))

    fishset = grid.ParticleSet(size=individuals, pclass=SEAPODYM_SKJ, start_field=grid.SEAPODYM_Density)

    age = fishset.Kernel(AgeParticle)
    diffuse = fishset.Kernel(LagrangianDiffusion)
    advect = fishset.Kernel(Advection)
    follow_gradient_rk4 = fishset.Kernel(GradientRK4)
    move = fishset.Kernel(Move)
    sampH = fishset.Kernel(SampleH)

    fishset.execute(sampH + age + follow_gradient_rk4 + advect + diffuse + move, endtime=fishset.grid.time[0]+time*timestep, dt=timestep,
                 output_file=fishset.ParticleFile(name=filename+"_results"),
                 output_interval=substeps, density_field=density_field)

    grid.write(args.output)


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
    p.add_argument('-f', '--files', default=['../Test_Data/SEAPODYM1997_PHYS_scaled.nc', '../Test_Data/SEAPODYM1997_PHYS_scaled.nc', '../Test_Data/SEAPODYM1997_HABITAT_depth3.nc'],
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
    p.add_argument('-s', '--startfield', type=str, default='none',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-o', '--output', default='habitat_grid',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=86400,
                   help='Length of timestep in seconds, defaults to one day')
    args = p.parse_args()
    filename = args.output

    U = args.files[0]
    V = args.files[1]
    H = args.files[2]
    if args.startfield == "none":
        args.startfield = None

    SIMPODYM(forcingU=U, forcingV=V, forcingH=H, startD=args.startfield,
             Uname=args.netcdf_vars[0], Vname=args.netcdf_vars[1], Hname=args.netcdf_vars[2],
             dimLat=args.diensions[0], dimLon=args.diensions[1], dimTime=args.diensions[2],
             individuals=args.particles, timestep=args.timestep, time=args.time, output_file=args.output)