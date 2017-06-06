from parcels import *
import numpy as np
from argparse import ArgumentParser
from glob import glob
from py import path
from datetime import timedelta, datetime
import calendar


def delaystart(particle, grid, time, dt):
    if time > particle.deployed:
        if particle.active > -1: # Has not beached or been recovered
            particle.active += 1
        if time > particle.deployed + grid.FAD_duration:
            particle.active = -1
        if particle.active < -1:
            particle.active -= 1


def delayedAdvectionRK4(particle, grid, time, dt):
    if particle.active > 0:
        u1 = grid.U[time, particle.lon, particle.lat]
        v1 = grid.V[time, particle.lon, particle.lat]
        lon1, lat1 = (particle.lon + u1*.5*dt, particle.lat + v1*.5*dt)
        u2, v2 = (grid.U[time + .5 * dt, lon1, lat1], grid.V[time + .5 * dt, lon1, lat1])
        lon2, lat2 = (particle.lon + u2*.5*dt, particle.lat + v2*.5*dt)
        u3, v3 = (grid.U[time + .5 * dt, lon2, lat2], grid.V[time + .5 * dt, lon2, lat2])
        lon3, lat3 = (particle.lon + u3*dt, particle.lat + v3*dt)
        u4, v4 = (grid.U[time + dt, lon3, lat3], grid.V[time + dt, lon3, lat3])
        particle.prev_lon = particle.lon
        particle.prev_lat = particle.lat
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * dt


def KillFAD(particle):
    if particle.active > 0:
        print("FAD hit model bounds at %s|%s!" % (particle.lon, particle.lat))
        particle.active = -1 * particle.active
        particle.lon = particle.prev_lon
        particle.lat = particle.prev_lat


def loadBRANgrid(Ufilenames, Vfilenames,
                  vars={'U': 'u', 'V': 'v'},
                  dims={'lat': 'lat', 'lon': 'lon', 'time': 'time'},
                  shift=0):
    filenames = {'U': Ufilenames,
                 'V': Vfilenames}
    print("loading grid files:")
    print(filenames)
    grid = Grid.from_netcdf(filenames, vars, dims)
    grid.U.time += shift
    grid.V.time += shift
    return grid


def createEvenStartingDistribution(grid, field='U', lon_range=[110, 290], lat_range=[-20,20]):
    data = np.zeros(np.shape(getattr(grid, field).data[[0,1],:,:]), dtype=np.float32)
    def isOcean(cell, lim=1e30):
        return 0 if cell > lim else 1
    lats = getattr(grid, field).lat
    lons = getattr(grid, field).lon

    for x in range(np.where(lons == lon_range[0])[0], np.where(lons == lon_range[1])[0]):
        for y in range(np.where(lats == lat_range[0])[0], np.where(lats == lat_range[1])[0]):
            data[0,y,x] = isOcean(getattr(grid, field).data[0,y,x])
    return Field('start', data, lons, lats, time=np.arange(2, dtype=np.float32))


def advanceGrid1Month(grid, gridnew):
    for v in grid.fields:
            vnew = getattr(gridnew, v.name)
            # Very roughly, we will kick out 30 days and load the new month (28-31 days)
            days = len(vnew.time)
            if np.min(vnew.time) > v.time[-1]:  # forward in time, so appending at end
                v.data = np.concatenate((v.data[days:, :, :], vnew.data[:, :, :]), 0)
                v.time = np.concatenate((v.time[days:], vnew.time))
            elif np.max(vnew.time) < v.time[0]:  # backward in time, so prepending at start
                v.data = np.concatenate((vnew.data[:, :, :], v.data[:-days, :, :]), 0)
                v.time = np.concatenate((vnew.time, v.time[:-days]))
            else:
                raise RuntimeError("Time of gridnew in grid.advancetime() overlaps with times in old grid")
    return days


def EvenFADRelease(filenames, variables, dimensions, fad_density,
                   timestep=21600, time=3, seed_timestep=7, start_time=757382400, start_field=None, start_dims=None,
                   output_file='FADRelease', mode='scipy', first_file_date=757382400, shift=0):

    days2secs = 24*60*60
    first_time  = datetime.fromtimestamp(start_time)
    end_time = datetime.fromtimestamp(start_time+((seed_timestep+1)*days2secs*time))
    year_start = year = first_time.year
    month_start = month = first_time.month
    year_end = end_time.year
    month_end = end_time.month
    T = (year_end-year_start)*12 - month_start + month_end + 1

    first_month_index = first_time - datetime.fromtimestamp(first_file_date)
    last_month_index = end_time - datetime.fromtimestamp(first_file_date)

    first_month_index = int(float(first_month_index.days)/365 * 12)# - 1
    last_month_index = int(float(last_month_index.days)/365 * 12) +1
    print("Earliest month index = %s" % first_month_index)
    print("Last month index = %s" % last_month_index)

    print("Loading first grid snapshot")
    grid = loadBRANgrid(filenames[0][first_month_index:(first_month_index+3)],
                        filenames[1][first_month_index:(first_month_index+3)],
                        variables, dimensions, shift)

    grid.add_constant("FAD_duration", 6*30*24*60*60)
    if start_field is not None:
        grid.add_field(Field.from_netcdf(name='start',filenames=start_field, dimensions=start_dims))

    StartField = createEvenStartingDistribution(grid, field='U' if start_field is None else 'start')
    StartField.write(output_file)
    deployment_cells = np.count_nonzero(StartField.data)
    print(np.count_nonzero(StartField.data))

    abs_seeding_timestep = seed_timestep*7*days2secs

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class FAD(ParticleClass):
        deployed = Variable('deployed', dtype=np.float32, to_write=True)
        active = Variable('active', dtype=np.float32, to_write=True)
        prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False)
        prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False)
        #recovered = Variable('recovered', dtype=np.float32, to_write=True)

    # The total number of required particles is:
    # the FAD density * the num of cells in the release region * the number seeding timesteps in the run
    total_FADs = fad_density * deployment_cells * time
    print("Total number of particles = %s" % total_FADs)

    fadset = ParticleSet.from_field(grid, pclass=FAD, start_field=StartField, size=total_FADs)
    results_file = ParticleFile(output_file + '_trajectories', fadset)

    print("Setting deployment times for all particles")
    for f in range(len(fadset.particles)):
        fadset.particles[f].deployed = start_time + int((f-1)/(fad_density*deployment_cells))*abs_seeding_timestep
        fadset.particles[f].active = 0

    print("Deployment time of first FAD seeding event: %s" % datetime.fromtimestamp(fadset.particles[0].deployed))

    def add_months(sourcedate,months=1):
        print(sourcedate)
        month = sourcedate.month - 1 + months
        year = int(sourcedate.year + month / 12 )
        month = month % 12 + 1
        day = min(sourcedate.day,calendar.monthrange(year,month)[1])
        return datetime.strptime(str(year)+'-'+str(month)+'-'+str(day), '%Y-%m-%d')

    times = [first_time]
    for t in range(1,T):
        times.append(add_months(times[-1]))
    Density_Time = np.array([(t - datetime(1970, 1, 1)).total_seconds() for t in times], dtype=np.float32)
    Density_Data = np.full([len(range(first_month_index, last_month_index)), grid.U.lat.size, grid.U.lon.size],-1, dtype=np.float64)
    print(Density_Time)
    print(np.shape(Density_Data))
    FAD_Density = Field('Density', Density_Data, lon=grid.U.lon, lat=grid.U.lat, time=Density_Time)

    print("Starting Sim")
    days = 31
    for m in range(first_month_index, last_month_index):
        print("Month %s" % m)
        start = grid.U.time[0]
        end = grid.U.time[0]+(days*24*60*60)
        print("Grid timeorigin = %s" % grid.U.time_origin)
        print("Executing from %s until %s, should be %s steps" %
              (datetime.fromtimestamp(start), datetime.fromtimestamp(end), (end-start)/timestep))
        fadset.execute(fadset.Kernel(delayedAdvectionRK4) + fadset.Kernel(delaystart),
                       starttime=grid.U.time[0], endtime=grid.U.time[0]+(30*24*60*60), dt=timestep,
                       output_file=results_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: KillFAD})
        print("density index = %s" % range(first_month_index, last_month_index)==m)
        #FAD_Density = Field('Density', np.full([2, grid.U.lat.size, grid.U.lon.size],-1, dtype=np.float64),
         #               lon=grid.U.lon, lat=grid.U.lat, time=np.array([start,end], dtype=np.float32))
        FAD_Density.data[m,:,:] = np.transpose(fadset.density())
        FAD_Density.write(output_file)
        days = advanceGrid1Month(grid, loadBRANgrid(filenames[0][m+3], filenames[1][m+3], variables, dimensions, shift))
        print("grid.data size: %s-%s-%s" % (np.shape(grid.U.data)))


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    #p.add_argument('-f', '--files', default=['/short/e14/jsp561/OFAM/ocean_u_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_v_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_phy_1993_01.nc'],
     #              help='List of NetCDF files to load')
    p.add_argument('-f', '--files', default=['/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_u_1993_01.TropPac.nc', '/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_v_1993_01.TropPac.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-t', '--time', type=int, default=3,
                   help='The total time of the simulation, expressed as number of seeding timesteps')
    p.add_argument('-d', '--density', type=int, default=1,
                   help='Density of FADs, per forcing field cell, released each release timestep')
    p.add_argument('-o', '--output', default='FADRelease',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=86400,
                   help='Length of timestep in seconds, defaults to one day')
    p.add_argument('-st', '--seed_time', type=int, default=7,
                   help='Frequency with which new FADs are seeded, in days')
    p.add_argument('-s', '--starttime', type=float, default=1041339600,
                   help='Time of tag release (seconds since 1-1-1970)')
    p.add_argument('-sf', '--startfield', type=str, default='none',
                   help='Name of a netcdf with dimensions and land mask to use in start field creation')
    p.add_argument('-sfd', '--startfield_dems', type=str, nargs=3, default=['latitude', 'longitude', 'time'],
                   help='Name start field netcdf dimensions')
    p.add_argument('-l', '--location', type=float, nargs=2, default=[180,0],
                   help='Release location (lon,lat)')
    p.add_argument('-r', '--raijin_run', type=str, default='False',
                   help='Raijin run boolean, defaults to False (true will overwrite filename locations etc.')
    args = p.parse_args()

    if args.raijin_run == 'True':
        print("Raijin Run")
        raijin_run = True
    else:
        raijin_run = False

    if raijin_run:
        output_filename = '/short/e14/jsp561/' + args.output
        shift = 0
        # Shifts to match BRAN time origin should no longer be required
        #shift = timedelta(days=365)*9
        #shift = shift.total_seconds()
    else:
        output_filename = args.output
        shift = 0

    U = args.files[0]
    V = args.files[1]

    if args.startfield is 'none':
        args.startfield = None

    filenames = {'U': args.files[0], 'V': args.files[1]}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'time': 'Time'}
    if not raijin_run:
        dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}

    if raijin_run:
        filenames = [sorted(glob(str(path.local("/g/data/gb6/BRAN/BRAN_2016/OFAM/ocean_u_*.nc")))),
                     sorted(glob(str(path.local("/g/data/gb6/BRAN/BRAN_2016/OFAM/ocean_v_*.nc"))))]
        args.startfield = glob(str(path.local('SEAPODYM_Forcing_Data/Latest/HABITAT/2003Run/INTERIM-NEMO-PISCES_skipjack_habitat_index_19970115.nc')))
        first_file = 757382400
    else:
        filenames = [sorted(glob(str(path.local("SEAPODYM_Forcing_Data/Latest/HABITAT/2003Run/2003run_PHYS_month*.nc")))),
                     sorted(glob(str(path.local("SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month*.nc"))))]
        first_file = 1041339600

    print(args.starttime)

    EvenFADRelease(filenames, variables, dimensions, fad_density=args.density,
                   start_field=args.startfield, start_dims=args.startfield_dims,
               timestep=args.timestep, time=args.time, seed_timestep=args.seed_time, start_time=args.starttime,
               output_file=output_filename, mode=args.mode,
               first_file_date=first_file, shift=shift)
