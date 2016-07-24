from parcels import *
from SEAPODYM_functions import *
from Behaviour import *
import numpy as np
import math
from argparse import ArgumentParser

def SIMPODYM(forcingU, forcingV, forcingH, startD=None,
             Uname='u', Vname='v', Hname='habitat', Dname='density',
             dimLon='lon', dimLat='lat',dimTime='time',
             individuals=100, timestep=172800, time=30, start_age=4,
             output_density=False, output_file="SIMPODYM"):

    filenames = {'U': forcingU, 'V': forcingV, 'H': forcingH, 'SEAPODYM_Density': startD}
    variables = {'U': Uname, 'V': Vname, 'H': Hname, 'SEAPODYM_Density': Dname}
    dimensions = {'lon': dimLon, 'lat': dimLat, 'time': dimTime}

    print("Creating Grid")
    grid = Grid.from_netcdf(filenames=filenames, variables=variables, dimensions=dimensions, vmax=200)
    print("Grid contains fields:")
    for f in grid.fields:
        print(f)

    if output_density:
        # Add a density field that will hold particle densities
        grid.add_field(Field('Density', np.full([grid.U.lon.size, grid.U.lat.size, grid.U.time.size],-1, dtype=np.float64),
                       grid.U.lon, grid.U.lat, depth=grid.U.depth, time=grid.U.time, transpose=True))
        density_field = grid.Density
    else:
        density_field = None

    # Offline calculate the 'diffusion' grid as a function of habitat
    print("Creating Diffusion Field")
    K = Create_SEAPODYM_Diffusion_Field(grid.H, 24*60*60)
    grid.add_field(K)

    # Offline calculation of the diffusion and basic habitat grid
    print("Calculating Gradient Fields")
    gradients = grid.H.gradient()
    for field in gradients:
        grid.add_field(field)
    K_gradients = grid.K.gradient()
    for field in K_gradients:
        grid.add_field(field)

    ParticleClass = JITParticle if args.mode == 'jit' else Particle
    SKJ = Create_Particle_Class(ParticleClass)

    #if startD is not None:
    #    print("Creating SEAPODYM_Density field for start conditions from %s" % startD)
    #    grid.add_field(Field.from_netcdf('SEAPODYM_Density', dimensions=dimensions, filenames=startD))

    fishset = grid.ParticleSet(size=individuals, pclass=SKJ, start_field=grid.SEAPODYM_Density)

    for p in fishset.particles:
        p.setAge(start_age)

    age = fishset.Kernel(AgeParticle)
    diffuse = fishset.Kernel(LagrangianDiffusion)
    advect = fishset.Kernel(Advection)
    follow_gradient_rk4 = fishset.Kernel(GradientRK4)
    move = fishset.Kernel(Move)
    sampH = fishset.Kernel(SampleH)

    print("Starting Sim")
    fishset.execute(sampH + age + follow_gradient_rk4 + advect + diffuse + move, endtime=fishset.grid.time[0]+time*timestep, dt=timestep,
                    output_file=fishset.ParticleFile(name=filename+"_results"),
                    interval=timestep)#, density_field=density_field)

    grid.write(args.output)


def Create_Particle_Class(type=JITParticle):

    class SEAPODYM_SKJ(type):
        user_vars = {'H': np.float32, 'monthly_age': np.float32, 'age': np.float32, 'fish': np.float32, 'Vmax': np.float32,
                     'Vx': np.float32, 'Vy': np.float32,
                     'Dx': np.float32, 'Dy': np.float32,
                     'Ax': np.float32, 'Ay': np.float32,
                     'Cx': np.float32, 'Cy': np.float32}
        monthly_age = 4
        age = 4.*30*24*60*60
        Vmax = 0.
        Dv_max = 0.
        ready4M = False

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(SEAPODYM_SKJ, self).__init__(*args, **kwargs)
            self.setAge(4.)
            self.fish = 100000

        def setAge(self, months):
            self.age = months*30*24*60*60
            self.monthly_age = int(self.age/(30*24*60*60))
            self.Vmax = V_max(self.monthly_age)

        def age_school(self, dt):
            self.age += dt
            if self.age - (self.monthly_age*30*24*60*60) > (30*24*60*60):
                self.monthly_age += 1
                self.Vmax = V_max(self.monthly_age)
                self.fish *= 1-Mortality(self.monthly_age, H=self.H)

    return SEAPODYM_SKJ


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
    p.add_argument('-f', '--files', default=['SEAPODYM_Forcing_Data/SEAPODYM1997_PHYS_Prepped.nc', 'SEAPODYM_Forcing_Data/SEAPODYM1997_PHYS_Prepped.nc', 'SEAPODYM_Forcing_Data/SEAPODYM1997_HABITAT_Prepped.nc'],
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
    p.add_argument('-s', '--startfield', type=str, default='SEAPODYM_Forcing_Data/SEAPODYM1997_DENSITY_Prepped.nc',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-o', '--output', default='SIMPODYM1997',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=86400,
                   help='Length of timestep in seconds, defaults to one day')
    args = p.parse_args()
    filename = args.output

    U = args.files[0]
    V = args.files[1]
    H = args.files[2]
    if args.startfield == "None":
        args.startfield = None

    SIMPODYM(forcingU=U, forcingV=V, forcingH=H, startD=args.startfield,
             Uname=args.netcdf_vars[0], Vname=args.netcdf_vars[1], Hname=args.netcdf_vars[2],
             dimLat=args.dimensions[0], dimLon=args.dimensions[1], dimTime=args.dimensions[2],
             individuals=args.particles, timestep=args.timestep, time=args.time, output_file=args.output)