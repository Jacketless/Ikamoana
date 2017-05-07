from parcels import *
from SEAPODYM_functions import *
from Behaviour import *
from glob import glob
import numpy as np
import math
from argparse import ArgumentParser


def IKAMOANA(Forcing_Files, Init_Distribution_File=None, Init_Positions=None, individuals=100, timestep=86400, T=30,
             Forcing_Variables={'H':'habitat'}, Forcing_Dimensions={'lon':'longitude', 'lat':'latitude', 'time':'time'},
             Kernels=['Advection_C', 'TaxisRK4', 'LagrangianDiffusion', 'Move'], mode='jit',
             start_age=4, output_density=False, output_grid=False, output_trajectory=True, random_seed=None,
             output_filestem='IKAMOANA'
             ):
    if random_seed is None:
        np.random.RandomState()
        random_seed = np.random.get_state()
    else:
        np.random.RandomState(random_seed)

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    Animal = define_Animal_Class(ParticleClass)

    # Create the forcing fieldset grid for the simulation
    ocean = Grid.from_netcdf(filenames=Forcing_Files, variables=Forcing_Variables, dimensions=Forcing_Dimensions,
                             vmin=-200, vmax=1e34, allow_time_extrapolation=True)

    if 'TaxisRK4' in Kernels:
        T_grads = ocean.H.gradient()
        ocean.add_field(T_grads[0])
        ocean.add_field(T_grads[1])
        t_fields = Create_SEAPODYM_Taxis_Fields(ocean.dH_dx, ocean.dH_dy, start_age)
        ocean.add_field(t_fields[0])
        ocean.add_field(t_fields[1])
    if 'LagrangianDiffusion' in Kernels:
        if not hasattr(ocean, 'dH_dx'):
            grads = ocean.H.gradient()
            ocean.add_field(grads[0])
            ocean.add_field(grads[1])
        ocean.add_field(Create_SEAPODYM_Diffusion_Field(ocean.H, timestep=timestep, start_age=start_age))
        K_grads = ocean.K.gradient()
        ocean.add_field(K_grads[0])
        ocean.add_field(K_grads[1])

    if 'MoveOffLand' in Kernels:
        ocean.add_field(Create_Landmask(ocean))

    if output_grid:
        ocean.write(output_filestem)

    # Create the particle set for our simulated animals
    animalset = ParticleSet.from_list(grid=ocean, lon=Init_Positions[0], lat=Init_Positions[1], pclass=Animal)
    trajectory_file = ParticleFile(output_filestem + '_trajectories', animalset) if output_trajectory is True else None

    # Build kernels
    AllKernels = {'Advection_C':Advection_C,
                  'LagrangianDiffusion':LagrangianDiffusion,
                  'TaxisRK4':TaxisRK4,
                  'Move': Move,
                  'MoveOffLand': MoveOffLand}
    KernelList = []
    KernelString = []
    for k in Kernels:
        print("Building kernel %s" % k)
        KernelList.append(animalset.Kernel(AllKernels[k]))
        KernelString.append(id(KernelList[-1]))
    # Concatenate for particleset execution (using eval is very ugly, but can't work out another way right now)
    KernelString = sum([['KernelList[',str(i),']','+'] for i in range(len(KernelList))], [])[:-1]
    KernelString = ''.join(KernelString)
    behaviours = eval(KernelString)

    # Execute
    start_time = getattr(ocean, Forcing_Variables.keys()[0]).time[0]
    animalset.execute(behaviours, starttime=start_time, endtime=start_time+timestep*T, dt=timestep,
                      output_file=trajectory_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: UndoMove})


def define_Animal_Class(type=JITParticle):
    class IKAMOANA_Animal(type):
        active = Variable("active", to_write=False)
        age = Variable('age',dtype=np.float32)
        Dx = Variable('Dx', to_write=True, dtype=np.float32)
        Dy = Variable('Dy', to_write=True, dtype=np.float32)
        Cx = Variable('Cx', to_write=True, dtype=np.float32)
        Cy = Variable('Cy', to_write=True, dtype=np.float32)
        Vx = Variable('Vx', to_write=True, dtype=np.float32)
        Vy = Variable('Vy', to_write=True, dtype=np.float32)
        Ax = Variable('Ax', to_write=True, dtype=np.float32)
        Ay = Variable('Ay', to_write=True, dtype=np.float32)
        prev_lon = Variable('prev_lon', to_write=False)
        prev_lat = Variable('prev_lat', to_write=False)
        #land_trigger = Variable('land_trigger', to_write=True)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(IKAMOANA_Animal, self).__init__(*args, **kwargs)
            self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0.
            self.active = 1
    return IKAMOANA_Animal


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('-f', '--files', default=['SimpleTest/SimpleCurrents.nc',
                                             'SimpleTest/SimpleCurrents.nc',
                                             'SimpleTest/SimpleHabitat.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-v', '--variables', default=['U','V','H'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u','v','habitat'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['latitude', 'longitude', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-m', '--map_dimensions', default=['lat', 'lon', 'time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-t', '--time', type=int, default=50,
                   help='Time to run simulation, in timesteps')
    p.add_argument('-s', '--startfield', type=str, default='None',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-sv', '--startfield_varname', default='None',
                   help='Name of the density variable in the startfield netcdf file')
    p.add_argument('-sp', '--start_point', nargs=2, default=[115,0],
                   help='start location to release all particles (overides startfield argument)')
    p.add_argument('-o', '--output', default='IKAMOANA',
                   help='Output filename stem')
    p.add_argument('-ts', '--timestep', type=int, default=172800,
                   help='Length of timestep in seconds, defaults to two days')
    p.add_argument('-wd', '--write_density', type=bool, default=True,
                   help='Flag to calculate monthly densities, defaults to true')
    p.add_argument('-wg', '--write_grid', type=bool, default=False,
                   help='Flag to write grid files to netcdf, defaults to false')
    p.add_argument('-wp', '--write_particles', type=str, default=True,
                   help='Flag to write particle trajectory to netcdf, defaults to true')
    p.add_argument('-sa', '--start_age', type=int, default=4,
                   help='Starting age of tuna, in months (defaults to 4)')
    p.add_argument('-k', '--kernels', nargs='+', type=str, default=['LagrangianDiffusion', 'TaxisRK4','Move', 'MoveOffLand'],
                   help='List of kernels from Behaviour.py to run, defaults to diffusion and taxis')
    args = p.parse_args()

    start_point = [[args.start_point[0]]*args.particles, [args.start_point[1]]*args.particles]

    for arg in [args.write_density, args.write_grid, args.write_particles]:
        arg = True if arg is 'True' else False

    if args.startfield == "None":
        args.startfield = None
    else:
        args.start_point = None

    args.write_particles = False if args.write_particles == 'False' else True

    ForcingFiles = {}
    ForcingVariables = {}
    for f in range(len(args.files)):
        ForcingFiles.update({args.variables[f]: args.files[f]})
        ForcingVariables.update({args.variables[f]: args.netcdf_vars[f]})


    IKAMOANA(Forcing_Files=ForcingFiles, Forcing_Variables=ForcingVariables, individuals=args.particles,
             T=args.time,timestep=args.timestep, Kernels=args.kernels,
             Init_Positions=start_point, mode=args.mode,
             output_grid=args.write_grid)