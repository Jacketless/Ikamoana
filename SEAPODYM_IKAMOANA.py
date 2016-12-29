from parcels import *
from SEAPODYM_functions import *
from Behaviour import *
from py import path
from glob import glob
import numpy as np
import math
from argparse import ArgumentParser


def SIMPLEDYM_SIM(Ufilestem, Vfilestem, Hfilestem, startD=None,
              Uname='u', Vname='v', Hname='habitat', Dname='density',
              dimLon='lon', dimLat='lat',dimTime='time',
              individuals=100, timestep=172800, months=3, start_age=4,
              output_density=False, output_file="SIMPODYM", write_grid=False,
              random_seed=None, mode='jit'):
    if random_seed is None:
        np.random.RandomState()
        random_seed = np.random.get_state()
    else:
        np.random.RandomState(random_seed)

    filenames = {'U': Ufilestem, 'V': Vfilestem, 'H': Hfilestem}
    variables = {'U': Uname, 'V': Vname, 'H': Hname}
    dimensions = {'lon': dimLon, 'lat': dimLat, 'time': dimTime}

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle
    SKJ = Create_Particle_Class(ParticleClass)


    print("Starting Sim")
    for m in range(months):
        print("Starting Month %s" % m)
        month_files = {}
        for v in filenames.keys():
            month_files.update({v: filenames[v] + '_month' + str(m+1) + '.nc'})
        grid = Grid.from_netcdf(filenames=month_files, variables=variables, dimensions=dimensions, vmax=300)
        if m == 0:
            grid.add_field(Field.from_netcdf('Start', dimensions={'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'data': 'skipjack_cohort_20021015_density_M0'},
                                             filenames=path.local(startD), vmax=1000))
        print("Creating Diffusion Field")
        K = Create_SEAPODYM_Diffusion_Field(grid.H, 30*24*60*60, start_age=start_age+m)
        grid.add_field(K)
        print("Calculating H Gradient Fields")
        gradients = grid.H.gradient()
        for field in gradients:
            grid.add_field(field)
        print("Calculating K Gradient Fields")
        K_gradients = grid.K.gradient()
        for field in K_gradients:
            grid.add_field(field)
        grid.K.interp_method = grid.dK_dx.interp_method = grid.dK_dy.interp_method = grid.H.interp_method = \
            grid.dH_dx.interp_method = grid.dH_dx.interp_method = grid.U.interp_method = grid.V.interp_method = 'nearest'

        if m == 0:
            fishset = grid.ParticleSet(size=individuals, pclass=SKJ, start_field=grid.Start)
            results_file = ParticleFile(output_file + '_results', fishset)

        age = fishset.Kernel(AgeParticle)
        diffuse = fishset.Kernel(LagrangianDiffusion)
        advect = fishset.Kernel(Advection_C)
        taxis = fishset.Kernel(GradientRK4_C)
        move = fishset.Kernel(Move)
        print(grid.time)
        fishset.execute(age + advect + taxis + diffuse + move, starttime=grid.time[0], endtime=grid.time[0]+30*24*60*60, dt=timestep,
                        output_file=results_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: UndoMove})


def Create_Particle_Class(type=JITParticle):

    class SEAPODYM_SKJ(type):
        active = Variable("active", to_write=False)
        monthly_age = Variable("monthly_age", dtype=np.int32)
        age = Variable('age', to_write=False)
        Vmax = Variable('Vmax', to_write=True)
        Dv_max = Variable('Dv_max', to_write=False)
        fish = Variable('fish', to_write=False)
        H = Variable('H', to_write=False)
        Dx = Variable('Dx', to_write=False)
        Dy = Variable('Dy', to_write=False)
        Cx = Variable('Cx', to_write=False)
        Cy = Variable('Cy', to_write=False)
        Vx = Variable('Vx', to_write=False)
        Vy = Variable('Vy', to_write=False)
        Ax = Variable('Ax', to_write=False)
        Ay = Variable('Ay', to_write=False)
        taxis_scale = Variable('taxis_scale', to_write=False)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(SEAPODYM_SKJ, self).__init__(*args, **kwargs)
            self.setAge(4.)
            self.fish = 100000
            self.H = self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0
            self.taxis_scale = 1
            self.active = 1

        def setAge(self, months):
            self.age = months*30*24*60*60
            self.monthly_age = int(self.age/(30*24*60*60))
            self.Vmax = V_max(self.monthly_age)

    return SEAPODYM_SKJ


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('-f', '--files', default=['SEAPODYM_Forcing_Data/PHYSICAL/2003Cohort_PHYS',
                                             'SEAPODYM_Forcing_Data/PHYSICAL/2003Cohort_PHYS',
                                             'SEAPODYM_Forcing_Data/HABITAT/2003Cohort_HABITAT'],
                   help='List of NetCDF files to load')
    p.add_argument('-v', '--variables', default=['U', 'V', 'H'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u', 'v', 'habitat'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['lat', 'lon', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-m', '--map_dimensions', default=['lat', 'lon', 'time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-t', '--time', type=int, default=3,
                   help='Time to run simulation, in months')
    p.add_argument('-s', '--startfield', type=str, default='SEAPODYM_Forcing_Data/2002_Fields/DENSITY/INTERIM-NEMO-PISCES_skipjack_cohort_20021015_density_M0_20030115.nc',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-o', '--output', default='SIMPODYM2003',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=172800,
                   help='Length of timestep in seconds, defaults to two days')
    p.add_argument('-wd', '--write_density', type=bool, default=True,
                   help='Flag to calculate monthly densities, defaults to true')
    p.add_argument('-wg', '--write_grid', type=bool, default=False,
                   help='Flag to write grid files to netcdf, defaults to false')

    args = p.parse_args()
    if args.startfield == "None":
        args.startfield = None

    SIMPLEDYM_SIM(Ufilestem=args.files[0], Vfilestem=args.files[1], Hfilestem=args.files[2], startD=args.startfield,
                  Uname=args.netcdf_vars[0], Vname=args.netcdf_vars[1], Hname=args.netcdf_vars[2],
                  dimLat=args.dimensions[0], dimLon=args.dimensions[1], dimTime=args.dimensions[2],
                  individuals=args.particles, timestep=args.timestep, months=args.time,
                  output_density=args.write_density, output_file=args.output, write_grid=args.write_grid, mode=args.mode)
