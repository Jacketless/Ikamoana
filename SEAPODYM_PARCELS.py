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
             startlons=None, startlats=None,
             dimLon='lon', dimLat='lat',dimTime='time',
             Kfile=None, dK_dxfile=None, dK_dyfile=None, dH_dxfile=None, dH_dyfile=None,
             diffusion_boost=0, diffusion_scale=1, taxis_scale=1,
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
        K = Create_SEAPODYM_Diffusion_Field(grid.H, 24*60*60, start_age=start_age,
                                            diffusion_scale=diffusion_scale, diffusion_boost=diffusion_boost)
        grid.add_field(K)
    else:
        print(Kfile[-3:])
        if Kfile[-3:] == 'dym':
            grid.add_field()
        else:
            grid.add_field(Field.from_netcdf('K', {'lon': 'nav_lon', 'lat': 'nav_lat', 'time': 'time_counter', 'data': 'K'}, glob(str(path.local(Kfile)))))


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

    grid.K.interp_method = grid.dK_dx.interp_method = grid.dK_dy.interp_method = grid.H.interp_method = \
        grid.dH_dx.interp_method = grid.dH_dx.interp_method = grid.U.interp_method = grid.V.interp_method = 'nearest'

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle
    SKJ = Create_Particle_Class(ParticleClass)

    if individuals == 0:
        # Run full SEAPODYM numbers (can be big!)
        area = np.zeros(np.shape(grid.U.data[0, :, :]), dtype=np.float32)
        U = grid.U
        V = grid.V
        dy = (V.lon[1] - V.lon[0])/V.units.to_target(1, V.lon[0], V.lat[0])
        for y in range(len(U.lat)):
            dx = (U.lon[1] - U.lon[0])/U.units.to_target(1, U.lon[0], U.lat[y])
            area[y, :] = dy * dx
        # Convert to km^2
        area /= 1000*1000
        # Total fish is density*area
        total_fish = np.sum(grid.SEAPODYM_Density.data * area)
        print('Total no. of fish in SEAPODYM run = %s' % total_fish)
        print('Assuming schools of 10,000 fish, total number of individuals = %s' % str(round(total_fish/10000)))
        individuals = raw_input('Please enter the number of school (leave blank for default):')
        if individuals is '':
            individuals = 1000

    fishset = grid.ParticleSet(size=individuals, pclass=SKJ, start_field=grid.SEAPODYM_Density,
                               lon=startlons, lat=startlats)

    for p in fishset.particles:
        p.setAge(start_age)
        p.taxis_scale = taxis_scale

    age = fishset.Kernel(AgeParticle)
    diffuse = fishset.Kernel(LagrangianDiffusion)
    advect = fishset.Kernel(Advection_C)
    taxis = fishset.Kernel(GradientRK4_C)
    move = fishset.Kernel(Move)
    sampH = fishset.Kernel(SampleH)

    month_steps = np.arange(grid.time[0], grid.time[0]+time*timestep, step=30*24*60*60)
    print("Starting Sim")

    for m in month_steps:
        print("Starting Month %s, day %s, time %s" %
              (str(np.where(month_steps == m)[0][0]), (m - grid.time[0])/(24*60*60), m - grid.time[0]))
        fishset.execute(age + advect + taxis + diffuse + move, starttime=m, endtime=m+30*24*60*60-1, dt=timestep,
                        output_file=fishset.ParticleFile(name=output_file+"_month"+str(np.where(month_steps == m)[0][0])),
                        interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: UndoMove})
        print("Finished at time %s" % (m+30*24*60*60-1-grid.time[0]))
        grid.Density.data[np.where(month_steps == m)[0][0],:,:] = np.transpose(fishset.density(relative=True))

    if write_grid:
        grid.write(output_file)

    #Write parameter file
    params = {"forcingU": forcingU, "forcingV": forcingV, "forcingH":forcingH, "startD":startD,
             "Uname":Uname, "Vname":Vname, "Hname":Hname, "Dname":Dname,
             "dimLon":dimLon, "dimLat":dimLat,"dimTime":dimTime,
             "Kfile":Kfile, "dK_dxfile":dK_dxfile, "dK_dyfile":dK_dyfile,
             "diffusion_boost":diffusion_boost, "diffusion_scale":diffusion_scale,
             "dH_dxfile":dH_dxfile, "dH_dyfile":dH_dyfile, "taxis_dampener":taxis_scale,
             "individuals":individuals, "timestep":timestep, "time":time, "start_age":start_age,
             "output_density":output_density, "output_file":output_file, "random_seed":random_seed, "mode":mode}
    param_file = open(output_file+"_parameters.txt", "w")
    for p, val in params.items():
        param_file.write("%s %s\n" % (p, val))
    param_file.close()


def Create_Particle_Class(type=JITParticle):

    class SEAPODYM_SKJ(type):
        active = Variable("active", to_write=False)
        monthly_age = Variable("monthly_age", dtype=np.int32)
        age = Variable('age', to_write=False)
        Vmax = Variable('Vmax', to_write=False)
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
            self.taxis_scale = 0
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
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int,  default=None,
                   help='Generate grid file with given dimensions')
    p.add_argument('-f', '--files', default=['SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc', 'SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc', 'SEAPODYM_Forcing_Data/2002_Fields/HABITAT/2003_HABITAT.nc'],
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
    p.add_argument('-s', '--startfield', type=str, default='SEAPODYM_Forcing_Data/2002_Fields/DENSITY/2003_Density.nc',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-o', '--output', default='SIMPODYM2003',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=172800,
                   help='Length of timestep in seconds, defaults to two days')
    p.add_argument('-b', '--boost', type=float, default=0,
                   help='Constant boost to diffusivity of tuna')
    p.add_argument('-ds', '--diffusion_scale', type=float, default=1,
                   help='Extra scale parameter to diffusivity of tuna')
    p.add_argument('-td', '--taxis_scale', type=float, default=1,
                   help='Constant scaler to taxis of tuna')
    p.add_argument('-sa', '--start_age', type=int, default=4,
                   help='Assumed start age of tuna cohort, in months. Defaults to 4')
    p.add_argument('-wd', '--write_density', type=bool, default=True,
                   help='Flag to calculate monthly densities, defaults to true')
    p.add_argument('-wg', '--write_grid', type=bool, default=False,
                   help='Flag to write grid files to netcdf, defaults to false')

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
             diffusion_boost=args.boost, diffusion_scale=args.diffusion_scale, taxis_scale=args.taxis_scale,
             start_age=args.start_age, individuals=args.particles, timestep=args.timestep, time=args.time,
             output_density=args.write_density, output_file=args.output, write_grid=args.write_grid, mode=args.mode)
