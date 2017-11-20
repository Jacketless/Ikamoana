from parcels import *
import numpy as np
from argparse import ArgumentParser
from glob import glob
from py import path
from datetime import timedelta, datetime
import calendar
import random


def delaystart(particle, fieldset, time, dt):
    if time > particle.deployed:
        if particle.active > -1: # Has not beached or been recovered
            particle.active += 1
        if time > particle.deployed + fieldset.FAD_duration:
            particle.active = -1
        if particle.active < -1:
            particle.active -= 1


def delayedAdvectionRK4(particle, fieldset, time, dt):
    if particle.active > 0:
        u1 = fieldset.U[time, particle.lon, particle.lat, particle.depth]
        v1 = fieldset.V[time, particle.lon, particle.lat, particle.depth]
        lon1, lat1 = (particle.lon + u1*.5*dt, particle.lat + v1*.5*dt)
        u2, v2 = (fieldset.U[time + .5 * dt, lon1, lat1, particle.depth], fieldset.V[time + .5 * dt, lon1, lat1, particle.depth])
        lon2, lat2 = (particle.lon + u2*.5*dt, particle.lat + v2*.5*dt)
        u3, v3 = (fieldset.U[time + .5 * dt, lon2, lat2, particle.depth], fieldset.V[time + .5 * dt, lon2, lat2, particle.depth])
        lon3, lat3 = (particle.lon + u3*dt, particle.lat + v3*dt)
        u4, v4 = (fieldset.U[time + dt, lon3, lat3, particle.depth], fieldset.V[time + dt, lon3, lat3, particle.depth])
        particle.prev_lon = particle.lon
        particle.prev_lat = particle.lat
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * dt


def KillFAD(particle, fieldset, time, dt):
    if particle.active > 0:
        #print("FAD hit model bounds at %s|%s!" % (particle.lon, particle.lat))
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
    grid = FieldSet.from_netcdf(filenames, vars, dims,
                                indices={'lat': range(250,1250), 'lon': range(1000, 2900), 'depth': range(0,9)})
    grid.U.time += shift
    grid.V.time += shift

    #integrate top depth layers
    depths = [0, 2.5, 7.5, 12.5, 17.515390396118164, 22.667020797729492,
         28.16938018798828, 34.2180061340332, 40.95497512817383, 48.45497512817383]
    weights = np.diff(depths)
    print("First layer sample = %s" % grid.U.data[0, 0, 500, 500])
    all_depths = grid.U.data[:, 0:len(weights), :, :]
    for d in range(len(weights)):
        all_depths[:, d, :, :] *= weights[d]
    grid.U.data[:, 0, :, :] = np.sum(all_depths, axis=1)/sum(weights)
    print("After weighting = %s" % grid.U.data[0, 0, 500, 500])

    print("First layer sample = %s" % grid.V.data[0, 0, 500, 500])
    all_depths = grid.V.data[:, 0:len(weights), :, :]
    for d in range(len(weights)):
        all_depths[:, d, :, :] *= weights[d]
    grid.V.data[:, 0, :, :] = np.sum(all_depths, axis=1) / sum(weights)
    print("After weighting = %s" % grid.V.data[0, 0, 500, 500])
    return grid


def createEvenStartingDistribution(grid, field='U', lon_range=[110, 290], lat_range=[-20,20]):
    if isinstance(field, basestring):
        data = np.zeros(np.shape(getattr(grid, field).data[[0,1],0,:,:]), dtype=np.float32)
        lats = getattr(grid, field).lat
        lons = getattr(grid, field).lon
    else:
        lons = np.round(np.arange(grid.U.lon[0], grid.U.lon[-1]+field, field, dtype=np.float32))
        lats = np.round(np.arange(grid.U.lat[0], grid.U.lat[-1]+field, field, dtype=np.float32))
        data = np.zeros([2, len(lats), len(lons)], dtype=np.float32)

    def isOcean(cell, lim=1e-30):
        return 0 if -1*lim < cell < lim else 1

    for x in range(np.where(lons == lon_range[0])[0], np.where(lons == lon_range[1])[0]):
        for y in range(np.where(lats == lat_range[0])[0], np.where(lats == lat_range[1])[0]):
            grid_x = np.where(np.round(grid.U.lon) == lons[x])[0]
            grid_y = np.where(np.round(grid.U.lat) == lats[y])[0]
            land_cells = []
            for gx in grid_x:
                for gy in grid_y:
                    land_cells.append(isOcean(grid.U.data[0,0,gy,gx]))
            data[0,y,x] = np.any(land_cells)
    return Field('start', data, lons, lats, time=np.arange(2, dtype=np.float32))


def advanceGrid1Month(grid, gridnew, days):
    for v in grid.fields:
            vnew = getattr(gridnew, v.name)
            if np.min(vnew.time) > v.time[-1]:  # forward in time, so appending at end
                v.data = np.concatenate((v.data[days:, :, :], vnew.data[:, :, :]), 0)
                v.time = np.concatenate((v.time[days:], vnew.time))
            elif np.max(vnew.time) < v.time[0]:  # backward in time, so prepending at start
                v.data = np.concatenate((vnew.data[:, :, :], v.data[:-days, :, :]), 0)
                v.time = np.concatenate((vnew.time, v.time[:-days]))
            else:
                raise RuntimeError("Time of gridnew in grid.advancetime() overlaps with times in old grid")
    days = len(vnew.time)
    return days


def EvenFADRelease(filenames, variables, dimensions, fad_density,
                   timestep=21600, time=3, seed_timestep=7, start_time=757382400, start_field_res=None, start_limits=[110, 290, -20, 20],
                   output_file='FADRelease', mode='scipy', first_file_date=757382400, shift=0,
                   write_density=True):

    days2secs = 24*60*60
    first_time  = datetime.fromtimestamp(start_time)
    end_time = datetime.fromtimestamp(start_time+((seed_timestep+1)*days2secs*time))
    print("Simulation starts %s, finishes %s" % (first_time, end_time))
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

    grid.add_constant("FAD_duration", 14*30*24*60*60)

    StartField = createEvenStartingDistribution(grid, field='U' if start_field_res is None else start_field_res,
                                                lon_range=[start_limits[0], start_limits[1]],
                                                lat_range=[start_limits[2], start_limits[3]])
    StartField.write(output_file)
    deployment_cells = np.count_nonzero(StartField.data)
    print("Number of cells in deployment zone = %s" %np.count_nonzero(StartField.data))

    abs_seeding_timestep = seed_timestep*days2secs

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class FAD(ParticleClass):
        deployed = Variable('deployed', dtype=np.float32, to_write=False)
        active = Variable('active', dtype=np.int32, to_write=True)
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
    fd_list = range(len(fadset.particles))
    random.shuffle(fd_list)
    for f in range(len(fadset.particles)):
        fadset.particles[f].deployed = start_time + np.floor((fd_list[f])/(fad_density*deployment_cells))*abs_seeding_timestep
        fadset.particles[f].active = 0
    print("Unique deployment dates are: %s" % [datetime.fromtimestamp(s) for s in set([f.deployed for f in fadset.particles])])
    print("Deployment time of first FAD seeding event: %s" % datetime.fromtimestamp(min([f.deployed for f in fadset.particles])))

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
    Density_Data = np.full([len(range(first_month_index, last_month_index)), StartField.lat.size, StartField.lon.size],-1, dtype=np.float64)
    print(Density_Time)
    print(np.shape(Density_Data))
    FAD_Density = Field('Density', Density_Data, lon=StartField.lon, lat=StartField.lat, time=Density_Time)

    print("Starting Sim")
    def last_day_of_month(any_day):
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
        return (next_month - datetime.timedelta(days=next_month.day)).day
    days = last_day_of_month(first_time)
    for m in range(first_month_index, last_month_index):
        print("Month %s" % m)
        start = grid.U.time[0]# + shift
        end = grid.U.time[0]+(days*24*60*60)
        print("Grid timeorigin = %s" % grid.U.time_origin)
        print("Executing from %s until %s, should be %s steps" %
              (datetime.fromtimestamp(start), datetime.fromtimestamp(end), (end-start)/timestep))
        fadset.execute(fadset.Kernel(delaystart) + fadset.Kernel(delayedAdvectionRK4),
                       starttime=grid.U.time[0], runtime=(days*24*60*60), dt=timestep,
                       output_file=results_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: KillFAD})
        if write_density:
            density_index = np.where([r == m for r in range(first_month_index, last_month_index)])[0]
            print("density index = %s" % density_index)
            FAD_Density.data[density_index,:,:] = np.transpose(fadset.density(StartField))
            FAD_Density.write(output_file)
        if m+3 < length(filenames): # As long as there are files for the new month, advance the grid
            days = advanceGrid1Month(grid, loadBRANgrid(filenames[0][m+3], filenames[1][m+3], variables, dimensions, shift), days)
        else: # Just cut off month during last advection loop
            grid.data = grid.data[days:,:,:]
            grid.time = grid.time[days:]
        print("grid.data size: %s-%s-%s-%s" % (np.shape(grid.U.data)))


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
    p.add_argument('-sfr', '--startfield_res', type=float, default=1,
                   help='Resolution in degrees for start field grid')
    p.add_argument('-l', '--location', type=float, nargs=2, default=[180,0],
                   help='Release location (lon,lat)')
    p.add_argument('-lm', '--limits', type=float, nargs=4, default=[110, 290, -20, 20],
                   help='Lon/lat limits for the even release start field')
    p.add_argument('-wd', '--write_density', type=str, default='False',
                   help='Boolean for whether to calculate and write density of FADs at runtime')
    p.add_argument('-r', '--raijin_run', type=str, default='False',
                   help='Raijin run boolean, defaults to False (true will overwrite filename locations etc.')
    args = p.parse_args()

    args.write_density = True if args.write_density is 'True' else False

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
    dimensions = {'lon': 'xu_ocean', 'lat': 'yu_ocean', 'time': 'Time', 'depth': 'st_ocean'}
    #if not raijin_run:
     #   dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}

    if raijin_run:
        filenames = [sorted(glob(str(path.local("/g/data/gb6/BRAN/BRAN_2016/OFAM/ocean_u_*.nc")))),
                     sorted(glob(str(path.local("/g/data/gb6/BRAN/BRAN_2016/OFAM/ocean_v_*.nc"))))]
        first_file = 757382400
        shift = 283996800
    else:
        filenames = [sorted(glob(str(path.local("SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month*.nc")))),
                     sorted(glob(str(path.local("SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month*.nc"))))]
        filenames = [str(path.local("/Volumes/SAMSUNG_4TB/Ocean_Model_Data/OFAM_month1/ocean_u_1993_01.TropPac.nc")),
                     str(path.local("/Volumes/SAMSUNG_4TB/Ocean_Model_Data/OFAM_month1/ocean_v_1993_01.TropPac.nc"))]
        first_file = 1041339600
        first_file = 725846400
        shift = 694224000

    print(args.starttime)
    print(filenames)
    EvenFADRelease(filenames, variables, dimensions, fad_density=args.density,
                   start_field_res=args.startfield_res,
               timestep=args.timestep, time=args.time, seed_timestep=args.seed_time, start_time=args.starttime,
               output_file=output_filename, mode=args.mode,
               first_file_date=first_file, shift=shift,
                   write_density=args.write_density)
