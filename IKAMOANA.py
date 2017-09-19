from parcels import *
from SEAPODYM_functions import *
from Behaviour import *
from glob import glob
import numpy as np
import math
from argparse import ArgumentParser
import datetime


def IKAMOANA(Forcing_Files, Init_Distribution_File=None, Init_Positions=None,
             individuals=100, timestep=86400, T=30, density_timestep=None,
             Forcing_Variables={'H':'habitat'}, Forcing_Dimensions={'lon':'longitude', 'lat':'latitude', 'time':'time'},
             Landmask_File=None, SEAPODYM_parfile=None,
             BehaviourMode='SEAPODYM', mode='jit',
             start_age=4, starttime=1042588800, output_density=False, output_grid=False, output_trajectory=True, random_seed=None,
             output_filestem='IKAMOANA'
             ):
    if random_seed is None:
        np.random.RandomState()
        random_seed = np.random.get_state()
    else:
        np.random.RandomState(random_seed)

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    if SEAPODYM_parfile is not None:
        SEAPODYM_Params = readSEAPODYM_parfile(SEAPODYM_parfile)

    Behaviours = {'SEAPODYM': ['Advection_C', 'TaxisRK4', 'LagrangianDiffusion', 'MoveWithLandCheck', 'AgeAnimal'],
                  'Drifter': ['Advection_C', 'Move']}
    Kernels = Behaviours[BehaviourMode]

    Animal = define_Animal_Class(ParticleClass)


    # Create the forcing fieldset grid for the simulation
    ocean = FieldSet.from_netcdf(filenames=Forcing_Files, variables=Forcing_Variables, dimensions=Forcing_Dimensions,
                             vmin=-200, vmax=1e5, allow_time_extrapolation=True)
    print(Forcing_Files['U'])
    Depthdata = Field.from_netcdf('u_bathy', dimensions={'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                                     filenames=Landmask_File, allow_time_extrapolation=True)
    Depthdata.name = 'bathy'
    ocean.add_field(Depthdata)
    ocean.add_field(Create_Landmask(ocean))

    if 'TaxisRK4' in Kernels:
        dHdx, dHdy = getGradient(ocean.H, ocean.LandMask) #ocean.H.gradient()
        ocean.add_field(dHdx)
        ocean.add_field(dHdy)
        t_fields = Create_SEAPODYM_Taxis_Fields(ocean.dH_dx, ocean.dH_dy, start_age=int(start_age),
                                                vmax_a=SEAPODYM_Params['MSS_species'], vmax_b=SEAPODYM_Params['MSS_size_slope'])
        ocean.add_field(t_fields[0])
        ocean.add_field(t_fields[1])
    if 'LagrangianDiffusion' in Kernels:
        if not hasattr(ocean, 'dH_dx'):
            dHdx, dHdy = getGradient(ocean.H, ocean.LandMask)
            ocean.add_field(dHdx)
            ocean.add_field(dHdy)
        ocean.add_field(Create_SEAPODYM_Diffusion_Field(ocean.H, timestep=30*24*60*60, start_age=int(start_age),
                                                        sigma=SEAPODYM_Params['sigma_species'],
                                                        c=SEAPODYM_Params['c_diff_fish']))
        dKdx, dKdy = getGradient(ocean.K, ocean.LandMask, False)
        ocean.add_field(dKdx)
        ocean.add_field(dKdy)
    if 'FishingMortality' in Kernels:
        ocean.add_field(Create_SEAPODYM_F_Field(ocean.E))

    if output_grid:
        ocean.write(output_filestem)

    # Create the particle set for our simulated animals
    if Init_Positions is not None:
        animalset = ParticleSet.from_list(fieldset=ocean, lon=Init_Positions[0], lat=Init_Positions[1], pclass=Animal)
    else:
        animalset = ParticleSet.from_field(fieldset=ocean, start_field=ocean.start, size=individuals, pclass=Animal)
    for a in animalset.particles:
        a.age = start_age*30*24*60*60
        a.monthly_age = start_age

    trajectory_file = ParticleFile(output_filestem + '_trajectories', animalset) if output_trajectory is True else None

    # Build kernels
    AllKernels = {'Advection_C':Advection_C,
                  'LagrangianDiffusion':LagrangianDiffusion,
                  'TaxisRK4':TaxisRK4,
                  'FishingMortality':FishingMortality,
                  'NaturalMortality':NaturalMortality,
                  'Move': Move,
                  'MoveOffLand': MoveOffLand,
                  'MoveWithLandCheck': MoveWithLandCheck,
                  'AgeAnimal':AgeAnimal}
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

    # Calculate an outer execution loop for particle density calculations
    start_time = (starttime-datetime.datetime(1970,1,1)).total_seconds()
    #start_time = starttime #getattr(ocean, Forcing_Variables.keys()[0]).time[0]
    sim_time = timestep*T
    if density_timestep > 0:
        outer_timestep = (density_timestep*timestep) % sim_time
        sim_steps = np.arange(start_time, start_time+sim_time+1, outer_timestep)
        Density_data = np.zeros((len(sim_steps)+1, len(ocean.U.lat),len(ocean.U.lon)))
        print(np.shape(Density_data))
        Density = Field('Density', Density_data, ocean.U.lon, ocean.U.lat, time=np.append(sim_steps,start_time+sim_time+timestep))
        Density.data[0,:,:] = animalset.density(field=ocean.U, particle_val='school')
    else:
        outer_timestep = sim_time
        sim_steps = np.arange(start_time, start_time+sim_time+1, outer_timestep)

    # Execute
    print(sim_steps)
    for step in sim_steps[:-1]:
        start = datetime.datetime.fromtimestamp(step).strftime('%Y-%m-%d %H:%M:%S')
        end = datetime.datetime.fromtimestamp(step+outer_timestep).strftime('%Y-%m-%d %H:%M:%S')
        print("Executing from %s (%s) to %s (%s), in steps of %s" % (start, step, end, step+outer_timestep, timestep))
        animalset.execute(behaviours, starttime=step, endtime=step+outer_timestep, dt=timestep,
                          output_file=trajectory_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: UndoMove})
        if density_timestep > 0:
            print(np.where(sim_steps == step)[0]+1)
            Density.data[np.where(sim_steps == step)[0]+1,:,:] = animalset.density(field=ocean.U, particle_val='school')
    if density_timestep > 0:
        Density.write(output_filestem)


def define_Animal_Class(type=JITParticle):
    class IKAMOANA_Animal(type):
        active = Variable("active", to_write=True)
        age = Variable('age',dtype=np.float32)
        monthly_age = Variable('monthly_age',dtype=np.float32)
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
        school = Variable('school', to_write=True, dtype=np.float32)
        depletionF = Variable('depletionF', to_write=True, dtype=np.float32)
        depletionN = Variable('depletionN', to_write=True, dtype=np.float32)
        In_Loop = Variable('In_Loop', to_write=True, dtype=np.float32)
        #land_trigger = Variable('land_trigger', to_write=True)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(IKAMOANA_Animal, self).__init__(*args, **kwargs)
            self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0.
            self.active = 1
            self.school = 1000
    return IKAMOANA_Animal


def readParameterFile(file):
    with open(file) as f:
        params = f.readlines()
    params = [p.strip() for p in params]
    Params = dict([p.split(" ") for p in params])
    conversion = dict({'none': None, 'None': None,
                       'true': True, 'True': True,
                       'false': False, 'False': False})
    print(conversion.keys())
    for p in Params.keys():
        if Params[p] in conversion.keys():
            print(p)
            Params[p] = conversion[Params[p]]
    numericals = ['individuals', 'time', 'timestep', 'density_timestep', 'start_age']
    for p in numericals:
        print(p)
        Params[p] = float(Params[p])
    return Params


def prepParameters(Params):
    all_file_links = {'U': 'ForcingU', 'V': 'ForcingV', 'H': 'ForcingH', 'Ubathy': 'ForcingUbathy','Vbathy': 'ForcingVbathy',
                      'start': 'StartDist', 'LandMask': 'LandMask_file', 'K': 'K_file', 'dK_dx': 'dKdx_file', 'dK_dy': 'dKdy_file',
                      'Tx': 'Tx_file', 'Ty': 'Ty_file', 'dH_dx': 'dHdx_file', 'dH_dy': 'dHdy_file'}
    for p in ['ForcingU', 'ForcingV', 'ForcingH', 'ForcingUbathy', 'ForcingVbathy',
              'StartDist', 'SEAPODYM_parfile', 'LandMask',
              'K_file', 'dKdx_file', 'dKdy_file',
              'Tx_file', 'Ty_file', 'dHdx_file', 'dHdy_file']:
        if p in Params.keys():
            Params[p] = Params['ForcingDir'] + Params[p]

    Params['ForcingFiles'] = {}
    for f in all_file_links.keys():
        if all_file_links[f] in Params.keys():
            Params['ForcingFiles'].update({f: Params[all_file_links[f]]})

    # Params['ForcingFiles'].update({'U': Params['ForcingU'],
    #                                'V': Params['ForcingV'],
    #                                'H': Params['ForcingH'],
    #                                'start': Params['StartDist'],
    #                                'Ubathy': Params['ForcingUbathy'],
    #                                'Vbathy': Params['ForcingVbathy']})
    Params['ForcingVariables'] = {}

    Params['ForcingVariables'].update({'U': Params['Uvar'],
                                       'V': Params['Vvar'],
                                       'H': Params['Hvar'],
                                       'start': Params['Startvar'],
                                       'Ubathy': Params['Ubathyvar'],
                                       'Vbathy': Params['Vbathyvar']})
    Params['start_time'] = datetime.datetime.strptime(Params['start_time'], '%Y-%M-%d')
    return Params


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('paramfile', default=None,
                   help='Parameter text file containing all parameters and file locations for Ikamoana run')
    p.add_argument('-m' '--mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-i', '--individuals', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('-f', '--files', default=['SimpleTest/SimpleCurrents.nc',
                                             'SimpleTest/SimpleCurrents.nc',
                                             'SimpleTest/SimpleHabitat.nc',
                                             'SimpleTest/SimpleMortality.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-fs', '--filestem', default=['n',
                                                 'n',
                                                 'n',
                                                 'n'],
                   help="Filestem options for each forcing file, defaulting to none for all")
    p.add_argument('-v', '--variables', default=['U','V','H','F'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u','v','habitat', 'F'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['latitude', 'longitude', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-md', '--map_dimensions', default=['lat', 'lon', 'time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-t', '--time', type=int, default=30,
                   help='Time to run simulation, in timesteps')
    p.add_argument('-st', '--start_time', type=int, default=1042588800,
                   help='Start time of simulation, in seconds since 1-1-1970, defaults to 15-1-2003')
    p.add_argument('-s', '--startfield', type=str, default='None',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-sv', '--startfield_varname', default='start',
                   help='Name of the density variable in the startfield netcdf file')
    p.add_argument('-sp', '--start_point', nargs='+', type=float, default=None,
                   help='start location to release all particles (overides startfield argument)')
    p.add_argument('-o', '--output', default='IKAMOANA',
                   help='Output filename stem')
    p.add_argument('-ts', '--timestep', type=int, default=172800,
                   help='Length of timestep in seconds, defaults to two days')
    p.add_argument('-dts', '--density_timestep', type=int, default=None,
                   help='Number of timesteps at which to calculate and write animal density, defaults to no density being written')
    p.add_argument('-wg', '--write_grid', type=bool, default=False,
                   help='Flag to write grid files to netcdf, defaults to false')
    p.add_argument('-wp', '--write_particles', type=str, default=True,
                   help='Flag to write particle trajectory to netcdf, defaults to true')
    p.add_argument('-sa', '--start_age', type=int, default=4,
                   help='Starting age of tuna, in months (defaults to 4)')
    p.add_argument('-k', '--kernels', nargs='+', type=str, default=['LagrangianDiffusion', 'TaxisRK4','Move', 'MoveOffLand', 'FishingMortality'],
                   help='List of kernels from Behaviour.py to run, defaults to diffusion and taxis')
    p.add_argument('-r', '--run', type=str, default='None',
                   help='Name of specific run (e.g. SEAPODYM_2003), overiding input files and variables')
    args = p.parse_args()

    for arg in [args.write_grid, args.write_particles]:
        arg = True if arg is 'True' else False

    if args.paramfile is not None:
        IKAMOANA_Args = readParameterFile(args.paramfile)
        IKAMOANA_Args = prepParameters(IKAMOANA_Args)
        print(IKAMOANA_Args)
    else:
        IKAMOANA_Args = {}
        IKAMOANA_Args['write_particles'] = False if args.write_particles == 'False' else True
        IKAMOANA_Args['Kernels'] = args.kernels
        IKAMOANA_Args['start_time'] = args.start_time
        IKAMOANA_Args['start_age'] = args.start_age

        IKAMOANA_Args['ForcingFiles'] = {}
        IKAMOANA_Args['ForcingVariables'] = {}
        for f in range(len(args.files)):
            IKAMOANA_Args['ForcingFiles'].update({args.variables[f]: args.files[f]})
            IKAMOANA_Args['ForcingVariables'].update({args.variables[f]: args.netcdf_vars[f]})
        if args.startfield == "None":
            IKAMOANA_Args['startfield'] = None
        else:
            IKAMOANA_Args['startpoint'] = None
            IKAMOANA_Args['ForcingFiles'].update({'start': args.startfield})
            IKAMOANA_Args['ForcingVariables'].update({'start': args.startfield_varname})

        if args.run is not 'None':
            temp_args = getSEAPODYMarguments(args.run)
            for arg in temp_args.keys():
                IKAMOANA_Args[arg] = temp_args[arg]
            #print(IKAMOANA_Args)

        if args.start_point is not None:
            IKAMOANA_Args['startpoint'] = [[],[]]
            points = len(args.start_point)/2
            for p in range(0,points*2, 2):
                IKAMOANA_Args['startpoint'][0].append([args.start_point[p]]*(args.particles/points))
                IKAMOANA_Args['startpoint'][1].append([args.start_point[p+1]]*(args.particles/points))
            IKAMOANA_Args['startpoint'][0] = [val for sublist in IKAMOANA_Args['startpoint'][0] for val in sublist]
            IKAMOANA_Args['startpoint'][1] = [val for sublist in IKAMOANA_Args['startpoint'][1] for val in sublist]


    IKAMOANA(Forcing_Files=IKAMOANA_Args['ForcingFiles'], Forcing_Variables=IKAMOANA_Args['ForcingVariables'], individuals=IKAMOANA_Args['individuals'],
             T=IKAMOANA_Args['time'],timestep=IKAMOANA_Args['timestep'], BehaviourMode=IKAMOANA_Args['Behaviour'], density_timestep=IKAMOANA_Args['density_timestep'],
             Landmask_File=IKAMOANA_Args['ForcingUbathy'], SEAPODYM_parfile=IKAMOANA_Args['SEAPODYM_parfile'],
             Init_Positions=None if 'startpoint' not in IKAMOANA_Args else IKAMOANA_Args['startpoint'], mode=IKAMOANA_Args['kernel_mode'],
             output_grid=IKAMOANA_Args['write_grid'], output_trajectory=IKAMOANA_Args['write_particles'],  output_filestem=args.output,
             starttime=IKAMOANA_Args['start_time'], start_age=IKAMOANA_Args['start_age'])