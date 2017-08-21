from parcels import *
from SEAPODYM_functions import *
from Behaviour import *
from glob import glob
import numpy as np
import math
from argparse import ArgumentParser


def SIMPLEDYM_SIM(Ufilestem, Vfilestem, Hfilestem, startD=None,
              Uname='u', Vname='v', Hname='habitat', Dname='skipjack_cohort_20021015_density_M0',
              dimLon='lon', dimLat='lat',dimTime='time',
              Kfilestem=None, field_units='m2_per_s',
              individuals=100, timestep=172800, months=3, start_age=4, start_month=1, start_year=2003, start_point=None,
              output_density=False, output_file="SIMPODYM", write_grid=False, write_trajectories=True,
              random_seed=None, mode='jit', verbose=True, filestem_simple={'H': True, 'U': False, 'V': False}):
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
    month_steps = np.arange(start_month, start_month+months)
    for m in month_steps:
        fish_age = start_age+np.where(month_steps == m)[0][0]
        print("Starting Month %s" % m)
        month_files = {}
        for v in filenames.keys():
            month_files.update({v: getForcingFilename(filenames[v], m, start_year=start_year, simple=filestem_simple[v])})
        if Kfilestem is not None:
            Kfile = (getForcingFilename(Kfilestem, m, start_year=start_year))
        else:
            Kfile = None
        if m == start_month:
            grid = Create_SEAPODYM_Grid(forcing_files=month_files, forcing_vars=variables, forcing_dims=dimensions,
                                        startD=startD, start_age=fish_age, verbose=verbose,
                                        diffusion_file=Kfile, field_units=field_units,
                                        startD_dims={'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'data': Dname})
            if verbose:
                total_pop = getPopFromDensityField(grid)
                print('Total no. of fish in SEAPODYM run = %s' % total_pop)
                print('therefor simulation assumes each particle represents %s individual fish' % str(total_pop/individuals))
            if start_point is not None:
                print("All particles starting at position %s %s" % (start_point[0], start_point[1]))
                fishset = grid.ParticleSet(size=individuals, pclass=SKJ,
                                           lon=[start_point[0]]*individuals, lat=[start_point[1]]*individuals)
            else:
                fishset = ParticleSet.from_field(grid, size=individuals, pclass=SKJ, start_field=grid.Start)
            results_file = ParticleFile(output_file + '_results', fishset) if write_trajectories is True else None
            if output_density:
                sim_time = np.linspace(grid.U.time[0], grid.U.time[0]+(months+1)*30*24*60*60, num=months+1)
                fish_density = Field('Density', np.full([grid.U.lon.size, grid.U.lat.size, sim_time.size],-1, dtype=np.float64),
                                     grid.U.lon, grid.U.lat, depth=grid.U.depth, transpose=True,
                                     time=sim_time)
                #Calculate density before initial timestep
                if output_density:
                    fish_density.data[np.where(month_steps == m)[0][0],:,:] = fishset.density()#np.transpose(fishset.density())
        else:
            grid = Create_SEAPODYM_Grid(forcing_files=month_files, forcing_vars=variables, forcing_dims=dimensions,
                                        start_age=fish_age, diffusion_file=Kfile, field_units=field_units, verbose=verbose)
            oldfishset = fishset
            fishset = ParticleSet.from_list(grid, pclass=SKJ, lon=[200]*individuals, lat=[0]*individuals)
            for p in range(len(fishset.particles)):
                Copy_Fish_Particle(oldfishset.particles[p], fishset.particles[p], SKJ)
                fishset.particles[p].In_Loop = 0


        if write_grid:
            grid.write(output_file + '_month' + str(m) + '_')

        for p in fishset.particles:
            p.setAge(fish_age)

        ## Test case (TO REMOVE)
        grid.add_constant('minlon', 160)
        grid.add_constant('maxlon', 240)
        grid.add_constant('minlat', -10)
        grid.add_constant('maxlat', 20)
        regionstop = fishset.Kernel(RegionBound)

        age = fishset.Kernel(AgeParticle)
        diffuse = fishset.Kernel(LagrangianDiffusion)
        advect = fishset.Kernel(Advection_C)
        taxis = fishset.Kernel(TaxisRK4)#fishset.Kernel(GradientRK4_C)
        #combinedadvection = fishset.Kernel(CurrentAndTaxisRK4)
        move = fishset.Kernel(MoveWithLandCheck)
        #landcheck = fishset.Kernel(MoveOffLand)
        sampH = fishset.Kernel(SampleH)
        moveeast = fishset.Kernel(MoveEast)
        movewest = fishset.Kernel(MoveWest)
        print("Executing kernels...")
        #end_time = 15*24*60*60 if m == start_month else 30*24*60*60
        run_time = 30*24*60*60
        fishset.execute(age + advect + taxis + diffuse + move + sampH, starttime=fishset[0].time, runtime=run_time, dt=timestep,
                        output_file=results_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: UndoMove})
        #fishset.execute(age + advect + taxis + diffuse + move + landcheck + sampH, starttime=grid.U.time[0], endtime=grid.U.time[0]+30*24*60*60, dt=timestep,
        #                output_file=results_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: UndoMove})
        #fishset.execute(movewest + landcheck, starttime=grid.time[0], endtime=grid.time[0]+30*24*60*60, dt=timestep,
         #               output_file=results_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: UndoMove})
        # Here we are calculating density AFTER particle execution each timestep
        if output_density:
            fish_density.data[np.where(month_steps == m)[0][0] + 1,:,:] = fishset.density()#np.transpose(fishset.density())
            fish_density.write(output_file)

    params = {"forcingU": Ufilestem, "forcingV": Vfilestem, "forcingH":Hfilestem, "startD":startD,
             "Uname":Uname, "Vname":Vname, "Hname":Hname, "Dname":Dname,
             "dimLon":dimLon, "dimLat":dimLat,"dimTime":dimTime,
             "Kfile":Kfilestem, "individuals":individuals, "timestep":timestep, "month_steps":months,
             "start_age":start_age, "start_month":start_month, "start_year":start_year, "trajectories_output":write_trajectories,
             "output_density":output_density, "output_file":output_file, "random_seed":random_seed, "mode":mode}
    write_parameter_file(params, output_file)


def getForcingFilename(stem, month_step, start_year=2003, simple=False):
    if simple:
        if month_step < 10:
            month_step = '0'+str(month_step)
        return(stem + str(month_step) + '.nc')
    else:
        year = int(start_year + math.floor((month_step-1)/12))
        month = month_step % 12
        if month == 0:
            month = 12
        if month < 10:
            month = '0' + str(month)
        return(stem + str(year) + str(month) + "15.nc")


def Create_Particle_Class(type=JITParticle):

    class SEAPODYM_SKJ(type):
        active = Variable("active", to_write=False)
        monthly_age = Variable("monthly_age", dtype=np.int32)
        age = Variable('age', to_write=False)
        #Vmax = Variable('Vmax', to_write=False, dtype=np.float32)
        #Dv_max = Variable('Dv_max', to_write=False)
        fish = Variable('fish', to_write=False)
        H = Variable('H', to_write=False, dtype=np.float32)
        dHdx = Variable('dHdx', to_write=False, dtype=np.float32)
        dHdy = Variable('dHdy', to_write=False, dtype=np.float32)
        # Dx = Variable('Dx', to_write=False, dtype=np.float32)
        # Dy = Variable('Dy', to_write=False, dtype=np.float32)
        # Cx = Variable('Cx', to_write=False, dtype=np.float32)
        # Cy = Variable('Cy', to_write=False, dtype=np.float32)
        # Vx = Variable('Vx', to_write=False, dtype=np.float32)
        # Vy = Variable('Vy', to_write=False, dtype=np.float32)
        # Ax = Variable('Ax', to_write=False, dtype=np.float32)
        # Ay = Variable('Ay', to_write=False, dtype=np.float32)
        In_Loop = Variable('In_Loop', to_write=True, dtype=np.float32)
        #taxis_scale = Variable('taxis_scale', to_write=False)
        prev_lon = Variable('prev_lon', to_write=True)
        prev_lat = Variable('prev_lat', to_write=True)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(SEAPODYM_SKJ, self).__init__(*args, **kwargs)
            self.setAge(4.)
            self.fish = 100000
            #self.H = self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0.0
            self.H = 0.0
            #self.taxis_scale = 1
            self.active = 1
            self.In_Loop = 0

        def setAge(self, months):
            self.age = months*30*24*60*60
            self.monthly_age = int(self.age/(30*24*60*60))
            #self.Vmax = V_max(self.monthly_age)

    return SEAPODYM_SKJ


def Copy_Fish_Particle(old_p, new_p, pclass):
    for v in pclass.getPType().variables:
        #print("Copying %s, oldvalue = %s, newvalue = %s" % (v.name, getattr(old_p, v.name), getattr(new_p, v.name)))
        setattr(new_p, v.name, getattr(old_p, v.name))

def write_parameter_file(params, file_stem):
    param_file = open(file_stem+"_parameters.txt", "w")
    for p, val in params.items():
        param_file.write("%s %s\n" % (p, val))
    param_file.close()


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('-r', '--run', default='2003')
    p.add_argument('-f', '--files', default=['SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month',
                                             'SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month',
                                             'SEAPODYM_Forcing_Data/Latest/HABITAT/2003Run/INTERIM-NEMO-PISCES_skipjack_habitat_index_'],
                   help='List of NetCDF files to load')
    p.add_argument('-v', '--variables', default=['U', 'V', 'H'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u', 'v', 'skipjack_habitat_index'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['latitude', 'longitude', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-m', '--map_dimensions', default=['lat', 'lon', 'time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-t', '--time', type=int, default=3,
                   help='Time to run simulation, in months')
    p.add_argument('-s', '--startfield', type=str, default='SEAPODYM_Forcing_Data/Latest/DENSITY/INTERIM-NEMO-PISCES_skipjack_cohort_20021015_density_M0_20030115.nc',
                   help='Particle density field with which to initiate particle positions')
    p.add_argument('-sv', '--startfield_varname', default='skipjack_cohort_20021015_density_M0',
                   help='Name of the density variable in the startfield netcdf file')
    p.add_argument('-sp', '--start_point', nargs=2, default=-1,
                   help='start location to release all particles (overides startfield argument)')
    p.add_argument('-k', '--k_file_stem', default='None',
                   help='File stem name for pre-calculated diffusivity fields')
    p.add_argument('-fu', '--field_file_units', default='m_per_s',
                   help='Units to use for calculating diffusion and taxis field data, defaults to meters (^2 for K) per second')
    p.add_argument('-o', '--output', default='SIMPLEDYM2003',
                   help='Output filename stem')
    p.add_argument('-ts', '--timestep', type=int, default=172800,
                   help='Length of timestep in seconds, defaults to two days')
    p.add_argument('-wd', '--write_density', type=str, default='True',
                   help='Flag to calculate monthly densities, defaults to true')
    p.add_argument('-wg', '--write_grid', type=bool, default=False,
                   help='Flag to write grid files to netcdf, defaults to false')
    p.add_argument('-wp', '--write_particles', type=str, default='True',
                   help='Flag to write particle trajectory to netcdf, defaults to true')
    p.add_argument('-sa', '--start_age', type=int, default=4,
                   help='Starting age of tuna, in months (defaults to 4)')
    p.add_argument('-sm', '--start_month', type=int, default=1,
                   help='Starting time of the simulation, in months, (defaults to month 1)')
    p.add_argument('-sy', '--start_year', type=int, default=2003,
                   help='Starting year of the simulation (defaults to 2003)')
    p.add_argument('-vb', '--verbose', type=bool, default=False,
                   help='Boolean for verbose print statements')
    args = p.parse_args()
    if args.startfield == "None":
        args.startfield = None
    if args.start_point == -1:
        args.start_point = None
    if args.k_file_stem == "None":
        args.k_file_stem = None
    args.write_particles = False if args.write_particles == 'False' else True
    args.write_density = False if args.write_density == 'False' else True

    if args.verbose != False:
        args.verbose = True

    # Dictionary for the timestep naming convention of files
    file_naming = {'H': False, 'U': True, 'V': True}

    if args.run == '1997':
        args.files = ['SEAPODYM_Forcing_Data/Latest/PHYSICAL/1997Run/1997run_PHYS_month',
                       'SEAPODYM_Forcing_Data/Latest/PHYSICAL/1997Run/1997run_PHYS_month',
                       'SEAPODYM_Forcing_Data/Latest/HABITAT/1997Run/INTERIM-NEMO-PISCES_skipjack_habitat_index_']
        args.start_year = 1997
        args.startfield = 'SEAPODYM_Forcing_Data/Latest/DENSITY/INTERIM-NEMO-PISCES_skipjack_cohort_19961015_density_M0_19970115.nc'
        args.startfield_varname = 'skipjack_cohort_19961015_density_M0'
    if args.run == 'test':
        args.files = ['TestCase/IdealTestCasePhysical_Month',
                       'TestCase/IdealTestCasePhysical_Month',
                       'TestCase/IdealTestCaseHabitat_Month']
        args.start_year = 2000
        args.startfield = 'TestCase/EvenStartDist.nc'
        args.startfield_varname = 'even_dist'
        file_naming = {'H': True, 'U': True, 'V': True}

    SIMPLEDYM_SIM(Ufilestem=args.files[0], Vfilestem=args.files[1], Hfilestem=args.files[2], startD=args.startfield,
                  Uname=args.netcdf_vars[0], Vname=args.netcdf_vars[1], Hname=args.netcdf_vars[2], Dname=args.startfield_varname,
                  dimLat=args.dimensions[0], dimLon=args.dimensions[1], dimTime=args.dimensions[2],
                  Kfilestem=args.k_file_stem, field_units=args.field_file_units,
                  individuals=args.particles, timestep=args.timestep, months=args.time,
                  start_age=args.start_age, start_month=args.start_month, start_year=args.start_year, start_point=args.start_point,
                  output_density=args.write_density, output_file=args.output, write_trajectories=args.write_particles,
                  write_grid=args.write_grid, mode=args.mode, filestem_simple=file_naming, verbose=args.verbose)
