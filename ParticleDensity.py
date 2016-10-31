from parcels import Field
from netCDF4 import Dataset
import numpy as np
from argparse import ArgumentParser


def calculateDensities(pfile, limits=[120, 290, -45, 45], p_var=None, output='Density'):
    print(pfile)
    particles = Dataset(pfile, 'r')
    lon = particles.variables['lon']
    lat = particles.variables['lat']
    #active = particles.variables['active']
    time = particles.variables['time']
    if p_var is not None:
        p_var = particles.variables[p_var]

    #lon = lon[range(0,10000), :]
    #lat = lat[range(0,10000), :]
    #active = particles.variables['active']
    #time = time[range(0,10000), :]

    class ParticleTrack(object):
        lon = []
        lat = []
        def __init__(self, lon, lat, active=1):
            self.lon = lon
            self.lat = lat
            # for t in range(len(lon)):
            #     #if active[t] == 1:
            #     self.pos[t,:,0] = lon[t]
            #     self.pos[t,0,:] = lat[t]

    Particles = []

    print('Shape of data')
    print(np.shape(lon))
    print(np.shape(lat))
    print(np.shape(time))

    #print('Making particles...')
    #for p in range(len(lon[:,0])):
    #    Particles.append(ParticleTrack(lon[p,:], lat[p,:]))#, active))

    lons = np.arange(limits[0], limits[1]+1)
    lats = np.arange(limits[2], limits[3]+1)
    month_time = np.arange(time[0,1], time[0,-1], 30*24*60*60)
    blank = np.zeros([len(month_time), len(lats), len(lons)], dtype=np.float32)

    DensityField = Field("TagDensity", blank, lon=lons, lat=lats,
                         time=month_time)
    print(np.shape(DensityField.data))
    print('Calculating Density...')

    for t in range(len(month_time)):
        mid_month = month_time[t] + (30*24*60*60)/2
        mid_month_i = np.argmin(abs(time[0,:] - mid_month))
        print(mid_month_i)
        DensityField.data[t, :, :] = np.transpose(density(DensityField, lon, lat, mid_month_i, particle_val=p_var,
                                                          relative=True, area_scale=False))

    DensityField.write('%s_density' % output)


def density(field, lons, lats, index = 0, particle_val=None, relative=False, area_scale=True):
    Density = np.zeros((field.lon.size, field.lat.size), dtype=np.float32)
    half_lon = (field.lon[1] - field.lon[0])/2
    half_lat = (field.lat[1] - field.lat[0])/2
    # Kick out particles that are not within the limits of our density field
    particles = (lons[:,index] > (np.min(field.lon)-half_lon)) * (lons[:,index] < (np.max(field.lon)+half_lon)) * \
                (lats[:,index] > (np.min(field.lat)-half_lat)) * (lats[:,index] < (np.max(field.lat)+half_lat))
    particles = np.where(particles)[0]
    print('Number of individuals = %s' % len(particles))
    # For each particle, find closest vertex in x and y and add 1 or val to the count
    if particle_val is not None:
        for p in particles:
            Density[np.argmin(np.abs(lons[p,index] - field.lon)), np.argmin(np.abs(lats[p,index] - field.lat))] \
                += particle_val[p,index]
    else:
        for p in particles:
            nearest_lon = np.argmin(np.abs(lons[p,index] - field.lon))
            nearest_lat = np.argmin(np.abs(lats[p,index] - field.lat))
            Density[nearest_lon, nearest_lat] += 1
        if relative:
            Density /= len(particles)

        if area_scale:
            area = np.zeros(np.shape(field.data[0, :, :]), dtype=np.float32)
            dx = (field.lon[1] - field.lon[0]) * 1852 * 60 * np.cos(field.lat*np.pi/180)
            dy = (field.lat[1] - field.lat[0]) * 1852 * 60
            for y in range(len(field.lat)):
                area[y, :] = dy * dx[y]
                # Scale by cell area
            Density /= np.transpose(area)

        #print(Density[np.where(Density > 0)])
        return Density


if __name__ == "__main__":
    p = ArgumentParser(description="""Calculation of density fields from particle trajectory NetCDFs""")
    p.add_argument('-p', '--pfile', default='none',
                   help='Particle file from which to calculate densities')
    p.add_argument('-l', '--limits', nargs=4, type=float, default=[120, 290, -45, 45],
                   help='Spatial lon/lat limits over which to calculate densities')
    p.add_argument('-v', '--variable', type=str, default='none',
                   help='Particle variable with which to calculate density (default is just the number of particles')
    p.add_argument('-o', '--output', default='DensityRatio',
                   help='Output file name')

    args = p.parse_args()

    if args.pfile == 'none':
        print("No particle file provided!")
    else:
        if args.variable is 'none':
            args.variable = None
        calculateDensities(args.pfile, args.limits, args.variable, args.output)
