from parcels import *
from Behaviour import *
import numpy as np
from SEAPODYM_functions import *
from argparse import ArgumentParser


def delaystart(particle, grid, time, dt):
    if time > particle.deployed:
        particle.active = 1


def delayedAdvectionRK4(particle, grid, time, dt):
    if particle.active == 1:
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


def FADRelease(grid, lons=[0], lats=[0], individuals=100, deploy_times=None, timestep=21600, time=30,
               output_file='FADRelease', mode='scipy'):

    starttime = grid.time[0]
    print("Start = %s (%s)" % (starttime, datetime.fromtimestamp(starttime)))

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class FAD(ParticleClass):
        deployed = Variable('deployed', dtype=np.float32)

    fadset = grid.ParticleSet(size=individuals, pclass=FAD, lon=lons, lat=lats)

    for f in range(len(fadset.particles)):
        fadset.particles[f].deployed = deploy_times[f]#-starttime#(deploy_times[f]-datetime.fromtimestamp(starttime+(22*365*24*60*60))).total_seconds()
        fadset.particles[f].active = -1
        print(fadset.particles[f].deployed)
        print(datetime.fromtimestamp(fadset.particles[f].deployed))

    print("Starting Sim")
    fadset.execute(fadset.Kernel(delayedAdvectionRK4) + fadset.Kernel(delaystart),
                    starttime=starttime, endtime=starttime+time*timestep, dt=timestep,
                    output_file=fadset.ParticleFile(name=output_file+"_results"),
                    interval=timestep)#, density_field=density_field)


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    #p.add_argument('-f', '--files', default=['/short/e14/jsp561/OFAM/ocean_u_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_v_1993_01.TropPac.nc', '/short/e14/jsp561/OFAM/ocean_phy_1993_01.nc'],
     #              help='List of NetCDF files to load')
    p.add_argument('-f', '--files', default=['/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_u_1993_01.TropPac.nc', '/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_v_1993_01.TropPac.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-de', '--deployment_file', default="Test_FAD_Deployments.txt",
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
    p.add_argument('-ts', '--timestep', type=int, default=3600,
                   help='Length of timestep in seconds, defaults to one day')
    p.add_argument('-s', '--starttime', type=int, default=None,
                   help='Time of tag release (seconds since 1-1-1970)')
    p.add_argument('-l', '--location', type=float, nargs=2, default=[180,0],
                   help='Release location (lon,lat)')
    args = p.parse_args()
    #filename = '/short/e14/jsp561/' + args.output

    U = args.files[0]
    V = args.files[1]

    filenames = {'U': args.files[0], 'V': args.files[1]}
    variables = {'U': args.netcdf_vars[0], 'V': args.netcdf_vars[1]}
    dimensions = {'lon': args.map_dimensions[1], 'lat': args.map_dimensions[0], 'time': args.map_dimensions[2]}

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

    times = times[0:10]

    first_time = datetime.fromtimestamp(np.min(times))
    last_time = datetime.fromtimestamp(np.max(times))

    print(first_time)
    print(last_time)

    year_start = year = first_time.year
    month_start = month = first_time.month
    year_end = last_time.year
    month_end = last_time.month

    T = (year_end-year_start)*12 - month_start + month_end + 1
    ufiles = []
    vfiles = []
    # create rest
    for t in range(T):
        if month < 10:
            ufiles.append("/g/data/gb6/BRAN/BRAN_3p5/OFAM/ocean_u_%s_0%s.nc" % (year, month))
            vfiles.append("/g/data/gb6/BRAN/BRAN_3p5/OFAM/ocean_v_%s_0%s.nc" % (year, month))
        else:
            ufiles.append("/g/data/gb6/BRAN/BRAN_3p5/OFAM/ocean_u_%s_%s.nc" % (year, month))
            vfiles.append("/g/data/gb6/BRAN/BRAN_3p5/OFAM/ocean_v_%s_%s.nc" % (year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    print(files)
    variables = {'U': args.netcdf_vars[0], 'V': args.netcdf_vars[1]}
    dimensions = {'lon': args.map_dimensions[1], 'lat': args.map_dimensions[0], 'time': args.map_dimensions[2]}

    grid = Grid.from_netcdf(filenames=files, variables=variables, dimensions=dimensions, vmin=-200, vmax=200)

    #shift = (datetime(1992,1,1) - datetime(1970,1,1)).total_seconds()
    #grid.time += shift

    #FADRelease(grid, lons=[float(v) for v in D_vars['"lon"']], lats=[float(v) for v in D_vars['"lat"']],
     #          deploy_times=times, individuals=N_FADs,
      #         timestep=args.timestep, time=args.time, output_file=args.output, mode=args.mode)
