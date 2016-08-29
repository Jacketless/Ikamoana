from parcels import *
from SEAPODYM_functions import *
from Behaviour import *
from py import path
from glob import glob
import numpy as np
import math
from argparse import ArgumentParser

def SIMPODYM(forcingU, forcingV, forcingH, startD=None,
             Uname='u', Vname='v', Hname='habitat', Dname='density',
             dimLon='lon', dimLat='lat',dimTime='time',
             Kfile=None, dK_dxfile=None, dK_dyfile=None, dH_dxfile=None, dH_dyfile=None,
             individuals=100, timestep=172800, time=30, start_age=4,
             output_density=False, output_file="SIMPODYM", write_grid=False,
             random_seed=None, mode='jit'):

    if random_seed is None:
        np.random.RandomState()
        random_seed = np.random.get_state()
    else:
        np.random.RandomState(random_seed)

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
    if Kfile is None:
        print("Creating Diffusion Field")
        K = Create_SEAPODYM_Diffusion_Field(grid.H, 24*60*60)
        grid.add_field(K)
    else:
        print(Kfile[-3:])
        if Kfile[-3:] == 'dym':
            grid.add_field(Field_from_DYM(Kfile, 'K'))
        else:
            grid.add_field(Field.from_netcdf('K', {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'K'}, glob(str(path.local(Kfile)))))

    print(dH_dxfile)
    print(dH_dyfile)
    if dH_dxfile is None or dH_dyfile is None:
        # Offline calculation of the diffusion and basic habitat grid
        print("Calculating H Gradient Fields")
        gradients = grid.H.gradient()
        for field in gradients:
            grid.add_field(field)
    else:
        print("Loading H Gradient Fields")
        grid.add_field(Field.from_netcdf('dH_dx', {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'dH_dx'}, glob(str(path.local(dH_dxfile)))))
        grid.add_field(Field.from_netcdf('dH_dy', {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'dH_dy'}, glob(str(path.local(dH_dyfile)))))

    if dK_dxfile is None or dK_dyfile is None:
        print("Calculating K Gradient Fields")
        K_gradients = grid.K.gradient()
        for field in K_gradients:
            grid.add_field(field)
    else:
        print("Loading K Gradient Fields")
        grid.add_field(Field.from_netcdf('dK_dx', {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'dK_dx'}, glob(str(path.local(dK_dxfile)))))
        grid.add_field(Field.from_netcdf('dK_dy', {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'dK_dy'}, glob(str(path.local(dK_dyfile)))))

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle
    SKJ = Create_Particle_Class(ParticleClass)

    #if startD is not None:
    #    print("Creating SEAPODYM_Density field for start conditions from %s" % startD)
    #    grid.add_field(Field.from_netcdf('SEAPODYM_Density', dimensions=dimensions, filenames=startD))

    fishset = grid.ParticleSet(size=individuals, pclass=SKJ, start_field=grid.SEAPODYM_Density)

    for p in fishset.particles:
        p.setAge(start_age)

    age = fishset.Kernel(AgeParticle)
    diffuse = fishset.Kernel(LagrangianDiffusion)
    advect = fishset.Kernel(Advection_C)
    taxis = fishset.Kernel(GradientRK4_C)
    move = fishset.Kernel(Move)
    sampH = fishset.Kernel(SampleH)

    print("Starting Sim")
    fishset.execute(age + advect + taxis + diffuse + move, endtime=fishset.grid.time[0]+time*timestep, dt=timestep,
                    output_file=fishset.ParticleFile(name=output_file+"_results"),
                    interval=timestep)#, density_field=density_field)
    if write_grid:
        grid.write(output_file)

    #Write parameter file
    params = {"forcingU": forcingU, "forcingV": forcingV, "forcingH":forcingH, "startD":startD,
             "Uname":Uname, "Vname":Vname, "Hname":Hname, "Dname":Dname,
             "dimLon":dimLon, "dimLat":dimLat,"dimTime":dimTime,
             "Kfile":Kfile, "dK_dxfile":dK_dxfile, "dK_dyfile":dK_dyfile, "dH_dxfile":dH_dxfile, "dH_dyfile":dH_dyfile,
             "individuals":individuals, "timestep":timestep, "time":time, "start_age":start_age,
             "output_density":output_density, "output_file":output_file, "random_seed":random_seed, "mode":mode}
    param_file = open(output_file+"_parameters.txt", "w")
    for p, val in params.items():
        param_file.write("%s %s\n" % (p, val))
    param_file.close()


def Create_Particle_Class(type=JITParticle):

    class SEAPODYM_SKJ(type):
        monthly_age = Variable("monthly_age", dtype=np.int32)
        age = Variable('age')
        Vmax = Variable('Vmax')
        Dv_max = Variable('Dv_max')
        fish = Variable('fish')
        H = Variable('H')
        Dx = Variable('Dx')
        Dy = Variable('Dy')
        Cx = Variable('Cx')
        Cy = Variable('Cy')
        Vx = Variable('Vx')
        Vy = Variable('Vy')
        Ax = Variable('Ax')
        Ay = Variable('Ay')

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(SEAPODYM_SKJ, self).__init__(*args, **kwargs)
            self.setAge(4.)
            self.fish = 100000
            self.H = self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0

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
    p.add_argument('-s', '--startfield', type=str, default='SEAPODYM_Forcing_Data/SEAPODYM2003_DENSITY_Prepped.nc',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-o', '--output', default='SIMPODYM2003',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=172800,
                   help='Length of timestep in seconds, defaults to two days')
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
             individuals=args.particles, timestep=args.timestep, time=args.time, output_file=args.output, mode=args.mode)
