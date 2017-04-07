from parcels import *
import numpy as np
from argparse import ArgumentParser
from glob import glob
from py import path
from datetime import timedelta, datetime


def delaystart(particle, grid, time, dt):
    if time > particle.deployed:
        if particle.active > -1: # Has not beached or been recovered
            particle.active += 1
        if time > particle.deployed + grid.FAD_duration:
            particle.active = -1
        if particle.active < -1:
            particle.active -= 1
        # if particle.active < 3: #Not beached
        #     particle.active = 1
        #     if time > particle.recovered:
        #         particle.active = 2


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
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * dt


def KillFAD(particle):
    print("FAD hit model bounds at %s|%s!" % (particle.lon, particle.lat))
    particle.active = -2
    particle.lon = 170
    particle.lat = -50


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


def advanceGrid1Month(grid, gridnew):
    for v in grid.fields:
            vnew = getattr(gridnew, v.name)
            # Very roughly, we will kick out 30 days and load the new month (28-31 days)
            days = 30# len(vnew.time)
            if np.min(vnew.time) > v.time[-1]:  # forward in time, so appending at end
                v.data = np.concatenate((v.data[days:, :, :], vnew.data[:, :, :]), 0)
                v.time = np.concatenate((v.time[days:], vnew.time))
            elif np.max(vnew.time) < v.time[0]:  # backward in time, so prepending at start
                v.data = np.concatenate((vnew.data[:, :, :], v.data[:-days, :, :]), 0)
                v.time = np.concatenate((vnew.time, v.time[:-days]))
            else:
                raise RuntimeError("Time of gridnew in grid.advancetime() overlaps with times in old grid")


def FADRelease(filenames, variables, dimensions, lons=[0], lats=[0], individuals=100, deploy_times=None, timestep=21600, time=30,
               output_file='FADRelease', mode='scipy', first_file_date=757382400, shift=0):

    first_time  = datetime.fromtimestamp(np.min(deploy_times))
    end_time = datetime.fromtimestamp(np.max(deploy_times))
    year_start = year = first_time.year
    month_start = month = first_time.month
    year_end = end_time.year
    month_end = end_time.month
    T = (year_end-year_start)*12 - month_start + month_end + 1

    print("Earliest time in forcing file list: %s" % datetime.fromtimestamp(first_file_date))
    print("Earliest deployment time in deployment file (index=%s): %s" % (np.where(deploy_times == np.min(deploy_times)), first_time))
    first_month_index = first_time - datetime.fromtimestamp(first_file_date)
    last_month_index = end_time - datetime.fromtimestamp(first_file_date)

    print(float(first_month_index.days)/365 * 12)

    first_month_index = int(float(first_month_index.days)/365 * 12) - 1
    last_month_index = int(float(last_month_index.days)/365 * 12)
    print("Earliest month index = %s" % first_month_index)
    print("Last month index = %s" % last_month_index)

    print("Loading first grid snapshot")
    grid = loadBRANgrid(filenames[0][first_month_index:(first_month_index+3)],
                        filenames[1][first_month_index:(first_month_index+3)],
                        variables, dimensions, shift)

    grid.add_constant("FAD_duration", 6*30*24*60*60)
    #grid.write('BRAN_test')

    #shift = (datetime(1992,1,1) - datetime(1970,1,1)).total_seconds()
    #grid.U.time += shift
    #grid.V.time += shift

    starttime = grid.U.time[0]
    print("Start = %s (%s)" % (starttime, datetime.fromtimestamp(starttime)))

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class FAD(ParticleClass):
        deployed = Variable('deployed', dtype=np.float32, to_write=False)
        active = Variable('active', dtype=np.float32, to_write=True)
        #recovered = Variable('recovered', dtype=np.float32, to_write=True)

    fadset = ParticleSet(grid, pclass=FAD, lon=lons, lat=lats)
    results_file = ParticleFile(output_file + '_trajectories', fadset)

    print("Setting deployment times for all particles")
    for f in range(len(fadset.particles)):
        fadset.particles[f].deployed = deploy_times[f]# + shift#-starttime#(deploy_times[f]-datetime.fromtimestamp(starttime+(22$
        fadset.particles[f].active = 0
        #fadset.particles[f].recovered = fadset.particles[f].deployed + shift + (timedelta(days=30)*6).total_seconds()
        #print(fadset.particles[f].deployed)
        #print(datetime.fromtimestamp(fadset.particles[f].deployed))
    print("Deployment time of first FAD in database: %s" % datetime.fromtimestamp(fadset.particles[0].deployed))

    print("Starting Sim")
    for m in range(first_month_index, last_month_index):
        print("Month %s" % m)
        start = grid.U.time[0]
        end = grid.U.time[0]+(30*24*60*60)
        print("Grid timeorigin = %s" % grid.U.time_origin)
        print("Executing from %s until %s, should be %s steps" %
              (datetime.fromtimestamp(start), datetime.fromtimestamp(end), (end-start)/timestep))
        fadset.execute(fadset.Kernel(delayedAdvectionRK4) + fadset.Kernel(delaystart),
                       starttime=grid.U.time[0], endtime=grid.U.time[0]+(30*24*60*60), dt=timestep,
                       output_file=results_file, interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: KillFAD})
        print("density index = %s" % range(first_month_index, last_month_index)==m)
        #FAD_Density = Field('Density', np.full([grid.U.lon.size, grid.U.lat.size, 1],-1, dtype=np.float64),
        #                grid.U.lon, grid.U.lat, depth=grid.U.depth, transpose=True,
        #                time=end)
        #FAD_Density.data[0,:,:] = np.transpose(fadset.density())
        #FAD_Density.write(output_file)
        advanceGrid1Month(grid, loadBRANgrid(filenames[0][m+3], filenames[1][m+3], variables, dimensions, shift))


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    #p.add_argument('-f', '--files', default=['/short/e14/jsp561/OFAM/ocean_u_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_v_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_phy_1993_01.nc'],
     #              help='List of NetCDF files to load')
    p.add_argument('-f', '--files', default=['/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_u_1993_01.TropPac.nc', '/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_v_1993_01.TropPac.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-de', '--deployment_file', default="Clean_TUBs_FAD_Deployments.txt",
                   help='List of NetCDF files to load')
    p.add_argument('-v', '--variables', default=['U', 'V'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u', 'v'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['lat', 'lon', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-m', '--map_dimensions', default=['yu_ocean', 'xu_ocean', 'Time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-t', '--time', type=int, default=800,
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-o', '--output', default='FADRelease',
                   help='List of NetCDF files to load')
    p.add_argument('-ts', '--timestep', type=int, default=86400,
                   help='Length of timestep in seconds, defaults to one day')
    p.add_argument('-s', '--starttime', type=int, default=None,
                   help='Time of tag release (seconds since 1-1-1970)')
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
        shift = timedelta(days=365)*9
        shift = shift.total_seconds()
    else:
        output_filename = args.output
        shift = 0

    U = args.files[0]
    V = args.files[1]

    filenames = {'U': args.files[0], 'V': args.files[1]}
    variables = {'U': args.netcdf_vars[0], 'V': args.netcdf_vars[1]}
    dimensions = {'lon': args.map_dimensions[1], 'lat': args.map_dimensions[0], 'time': args.map_dimensions[2]}
    if not raijin_run:
        dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    # Load FAD deployment data
    D_file = open(args.deployment_file, 'r')
    vars = D_file.readline().split()
    D_vars = {}
    print("Deployment file contains:")
    for v in vars:
        #print("- %s" % v)
        D_vars.update({v: []})
    for line in D_file:
        for v in range(len(vars)):
            D_vars[vars[v]].append(line.split()[v])

    N_FADs = len(D_vars['"time"'])
    print("Loaded %s FAD deployments" % N_FADs)

    times = [float(v) for v in D_vars['"time"']]

    if raijin_run:
        filenames = [sorted(glob(str(path.local("/g/data/gb6/BRAN/BRAN_2016/OFAM/ocean_u_*.nc")))),
                     sorted(glob(str(path.local("/g/data/gb6/BRAN/BRAN_2016/OFAM/ocean_v_*.nc"))))]
        first_file = 757382400
    else:
        filenames = [sorted(glob(str(path.local("SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003run_PHYS_month*.nc")))),
                     sorted(glob(str(path.local("SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003run_PHYS_month*.nc"))))]
        first_file = 1041339600
            #filenames = [glob(str(path.local('SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003run_PHYS_month*.nc')))] * 2
            #dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}


    #files = {'U': ufiles, 'V': vfiles}
    #files = {'U': "/g/data/gb6/BRAN/BRAN3p5/OFAM/ocean_u_*.nc", 'V': "/g/data/gb6/BRAN/BRAN3p5/OFAM/ocean_v_*.nc"}

    #print(filenames)

    FADRelease(filenames, variables, dimensions, lons=[float(v) for v in D_vars['"lon"']], lats=[float(v) for v in D_vars['"lat"']],
               deploy_times=times, individuals=N_FADs,
               timestep=args.timestep, time=args.time, output_file=output_filename, mode=args.mode,
               first_file_date=first_file, shift=shift)
