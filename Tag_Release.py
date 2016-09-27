from parcels import *
from Behaviour import *
import numpy as np
from SEAPODYM_functions import *
from SEAPODYM_PARCELS import *
from argparse import ArgumentParser


def TagRelease(grid, location=[0, 0], spread = 0, individuals=100, start=None, timestep=86400, time=30,
               output_file='TagRelease', mode='scipy', start_age=4):

    ParticleClass = JITParticle if mode == 'jit' else Particle

    class Tuna(ParticleClass):
        user_vars = {'monthly_age': np.float32, 'age': np.float32,
                     'displacement': np.float32}

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(Tuna, self).__init__(*args, **kwargs)
            self.monthly_age = 4
            self.age = self.monthly_age*30*24*60*60
            self.Vmax = V_max(self.monthly_age)
            self.startpos = [0, 0]
            self.displacement = 0

        def age_school(self, dt):
            self.age += dt
            if self.age - (self.monthly_age*30*24*60*60) > (30*24*60*60):
                self.monthly_age += 1
                self.Vmax = V_max(self.monthly_age)

        def update(self):
            self.displacement = np.sqrt(np.power(self.lon-self.startpos[0], 2) + np.power(self.lat-self.startpos[1], 2))

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle
    SKJ = Create_Particle_Class(ParticleClass)
    lats = []
    lons = []
    for _ in range(individuals):
        lats.append(location[1] + np.random.uniform(spread*-1, spread))
        lons.append(location[0] + np.random.uniform(spread*-1, spread))
    print(lats)
    print(lons)
    fishset = grid.ParticleSet(size=individuals, pclass=SKJ, lon=lons, lat=lats)

    age = fishset.Kernel(AgeParticle)
    diffuse = fishset.Kernel(LagrangianDiffusion)
    advect = fishset.Kernel(Advection_C)
    taxis = fishset.Kernel(GradientRK4_C)
    move = fishset.Kernel(Move)

    starttime = grid.time[0] if start is None else start
    if start is not None:
        for p in fishset.particles:
            p.age += starttime - grid.time[0]
            p.monthly_age = np.floor(p.age/30/24/60/60)
            p.Vmax = V_max(p.monthly_age)

    grid.write(output_file)

    print("Starting Sim")
    fishset.execute(age + taxis + advect + diffuse + move,
                    starttime=starttime, endtime=starttime+time*timestep, dt=timestep,
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
    p.add_argument('-s', '--starttime', type=int, default=None,
                   help='Time of tag release (seconds since 1-1-1970)')
    p.add_argument('-l', '--location', type=float, nargs=2, default=[180,0],
                   help='Release location (lon,lat)')
    p.add_argument('-a', '--age', type=int, default=4,
                   help='Starting age of release')
    p.add_argument('-c', '--cluster', type=float, default = 0.1,
                   help='+- spatial range around release location (uniformly distributed, in degrees lat/lon)')

    args = p.parse_args()
    filename = args.output

    U = args.files[0]
    V = args.files[1]
    H = args.files[2]

    grid = Create_SEAPODYM_Grid(forcingU=U, forcingV=V, forcingH=H,
             Uname=args.netcdf_vars[0], Vname=args.netcdf_vars[1], Hname=args.netcdf_vars[2],
             dimLat=args.dimensions[0], dimLon=args.dimensions[1], dimTime=args.dimensions[2],
                                start_age=11)
    print("Calculating H Gradient Fields")
    gradients = grid.H.gradient()
    for field in gradients:
        grid.add_field(field)
    print("Calculating K Gradient Fields")
    K_gradients = grid.K.gradient()
    for field in K_gradients:
        grid.add_field(field)

    TagRelease(grid, location=args.location, spread=args.cluster, start=args.starttime, individuals=args.particles,
               start_age=args.age, timestep=args.timestep, time=args.time, output_file=args.output, mode=args.mode)
