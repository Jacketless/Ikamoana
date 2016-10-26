import numpy as np
import struct
from parcels.field import Field
from parcels.grid import Grid
from parcels.particle import *
from netCDF4 import num2date
from datetime import datetime

def V_max(monthly_age, a=0.7343607395421234, b=0.5006692114850767):
    L = GetLengthFromAge(monthly_age)
    V = a * np.power(L, b)
    return V


def V_max_C(monthly_age):
    a=0.7343607395421234
    b=0.5006692114850767
    L = GetLengthFromAge(monthly_age)
    V = a * math.pow(L, b)
    return V


def GetLengthFromAge(monthly_age):
    # Linf = 88.317
    # K = 0.1965
    # return Linf * (1 - np.exp(-K * age))
    # Hard code discrete age-lengths for now
    lengths = [3.00, 4.51, 6.02, 11.65, 16.91, 21.83, 26.43, 30.72, 34.73, 38.49,
               41.99, 45.27, 48.33, 51.19, 53.86, 56.36, 58.70, 60.88, 62.92, 64.83,
               66.61, 68.27, 69.83, 71.28, 72.64, 73.91, 75.10, 76.21, 77.25, 78.22,
               79.12, 79.97, 80.76, 81.50, 82.19, 82.83, 83.44, 84.00, 84.53, 85.02,
               85.48, 85.91, 86.31, 86.69, 87.04, 87.37, 87.68, 87.96, 89.42]
    if monthly_age > 49:
        return 89.42/100
    else:
        return lengths[monthly_age-1]/100 # Convert to meters


def Mortality(age, MPmax=0.3, MPexp=0.1008314958945224, MSmax=0.006109001382111822, MSslope=0.8158285706493162,
              Mrange=1.430156372206337e-05, H=1):
    Mnat = MPmax*np.exp(-MPexp*age) + MSmax*np.power(age, MSslope)
    Mvar = Mnat * np.power(1 - Mrange, 1-H/2)
    return Mvar

def Mortality_C(age, H):
    MPmax=0.3
    MPexp=0.1008314958945224
    MSmax=0.006109001382111822
    MSslope=0.8158285706493162
    Mrange=1.430156372206337e-05
    Mnat = MPmax*math.exp(-MPexp*age) + MSmax*math.pow(age, MSslope)
    Mvar = Mnat * math.pow(1 - Mrange, 1-H/2)
    return Mvar


def Create_SEAPODYM_Diffusion_Field(H, timestep=86400, sigma=0.1999858740340303, c=0.9817751085550976, P=3,
                                    start_age=4, Vmax_slope=1, diffusion_boost=0, diffusion_scale=1, units='m_per_s'):
    K = np.zeros(np.shape(H.data), dtype=np.float32)
    months = start_age
    age = months*30*24*60*60
    for t in range(H.time.size):
        # Increase age in months if required, to incorporate appropriate Vmax
        age = H.time[t] - H.time[0] + start_age*30*24*60*60
        if age - (months*30*24*60*60) > (30*24*60*60):
            months += 1
        print("Calculating diffusivity for fish aged %s months" % months)
        if units == 'nm_per_mon':
            Dmax = np.power(GetLengthFromAge(months)*30*24*3600/1852, 2) / (4 * timestep/(60*60*24*30)) #vmax = L for diffusion
        else:
            Dmax = np.power(GetLengthFromAge(months), 2) / 4 * timestep #fixed b parameter for diffusion
        sig_D = sigma * Dmax
        for x in range(H.lon.size):
            for y in range(H.lat.size):
                K[t, y, x] = sig_D * (1 - c * np.power(H.data[t, y, x], P)) * diffusion_scale + diffusion_boost

    return Field('K', K, H.lon, H.lat, time=H.time)


def Create_SEAPODYM_Grid(forcingU, forcingV, forcingH, startD=None,
                         Uname='u', Vname='v', Hname='habitat', Dname='density',
                         dimLon='lon', dimLat='lat', dimTime='time', timestep=86400,
                         scaleH=None, start_age=4, output_density=False, diffusion_scale=1):
    filenames = {'U': forcingU, 'V': forcingV, 'H': forcingH}
    variables = {'U': Uname, 'V': Vname, 'H': Hname}
    dimensions = {'lon': dimLon, 'lat': dimLat, 'time': dimTime}

    if startD is not None:
        filenames.update({'SEAPODYM_Density': startD})
        variables.update({'SEAPODYM_Density': Dname})

    print("Creating Grid")
    grid = Grid.from_netcdf(filenames=filenames, variables=variables, dimensions=dimensions, vmin=-200, vmax=200)
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

    # Scale the H field between zero and one if required
    if scaleH is not None:
        grid.H.data /= np.max(grid.H.data)
        grid.H.data[np.where(grid.H.data < 0)] = 0
        grid.H.data *= scaleH

    # Offline calculate the 'diffusion' grid as a function of habitat
    print("Creating Diffusion Field")
    K = Create_SEAPODYM_Diffusion_Field(grid.H, timestep, start_age, diffusion_scale=diffusion_scale)
    grid.add_field(K)

    return grid


def Field_from_DYM(filename, name=None, xlim=None, ylim=None, fromyear=None, frommonth=0, toyear=None, tomonth=0):

    if name is None:
        name = str.split(filename, '/')[-1]
    print("Name = %s" % name)

    def lat_to_j(lat, latmax, deltaY):
        j = (int) ((latmax - lat)/deltaY + 1.5)
        return j-1

    def lon_to_i(lon, lonmin, deltaX):
        if lon < 0:
            lon = lon+360
        i = (int) ((lon - lonmin)/deltaX + 1.5)
        return i-1

    def get_tcount_start(zlevel, nlevel, date):
        n = 0
        while date > zlevel[n] and n < (nlevel-1):
            n += 1
        return n

    def get_tcount_end(zlevel, nlevel, date):
        n = nlevel-1
        while date < zlevel[n] and n > 0:
            n -= 1
        return n

    class DymReader:
        # Map well-known type names into struct format characters.
        typeNames = {
        'int32'  :'i',
        'uint32' :'I',
        'int64'  :'q',
        'uint64' :'Q',
        'float'  :'f',
        'double' :'d',
        'char'   :'s'}

        DymInputSize = 4

        def __init__(self, fileName):
            self.file = open(fileName, 'rb')

        def read(self, typeName):
            typeFormat = DymReader.typeNames[typeName.lower()]
            scale = 1
            if(typeFormat is 's'):
                scale = self.DymInputSize
            typeSize = struct.calcsize(typeFormat) * scale
            value = self.file.read(typeSize)
            decoded = struct.unpack(typeFormat*scale, value)
            #print(decoded)
            decoded = [x for x in decoded]

            if(typeFormat is 's'):
                decoded = ''.join(decoded)
                return decoded
            return decoded[0]

        def move(self, pos):
            self.file.seek(pos, 1)

        def close(self):
            self.file.close()

    if xlim is not None:
        x1 = xlim[0]
        x2 = xlim[1]
    else:
        x1 = x2 = 0

    if ylim is not None:
        y1 = ylim[0]
        y2 = ylim[1]
    else:
        y1 = y2 = 0

    file = DymReader(filename)

    # Get header
    print("-- Reading .dym file --")
    idformat = file.read('char')
    print("ID Format = %s" % idformat)
    idfunc = file.read('int32')
    print("IF Function = %s" % idfunc)
    minval = file.read('float')
    print("minval = %s" % minval)
    maxval = file.read('float')
    print("maxval = %s" % maxval)
    nlon = file.read('int32')
    print("nlon = %s" % nlon)
    nlat = file.read('int32')
    print("nlat = %s" % nlat)
    nlevel = file.read('int32')
    print("nlevel = %s" % nlevel)
    startdate = file.read('float')
    print("startdate = %s" % startdate)
    enddate = file.read('float')
    print("enddate = %s" % enddate)

    if fromyear is None:
        fromyear = np.floor(startdate)
    if toyear is None:
        toyear = np.floor(enddate)

    x = np.zeros([nlat, nlon], dtype=np.float32)
    y = np.zeros([nlat, nlon], dtype=np.float32)

    for i in range(nlat):
        for j in range(nlon):
            x[i, j] = file.read('float')

    for i in range(nlat):
        for j in range(nlon):
            y[i, j] = file.read('float')

    dx = x[0, 1] - x[0, 0]
    dy = y[0, 0] - y[1, 0]

    i1 = lon_to_i(x1, x[0, 0], dx)
    i2 = lon_to_i(x2, x[0, 0], dx)
    j1 = lat_to_j(y2, y[0, 0], dy)
    j2 = lat_to_j(y1, y[0, 0], dy)
    if xlim is None:
        i1 = 0
        i2 = nlon
    if ylim is None:
        j1 = 0
        j2 = nlat
    nlon_new = i2 - i1
    nlat_new = j2 - j1

    if xlim is None:
        nlon_new = nlon
        nlat_new = nlat
        i1 = 0
        i2 = nlon
        j1 = 0
        j2 = nlat

    for j in range(nlat_new):
        for i in range(nlon_new):
            x[j, i] = x[j+j1, i+i1]
            y[j, i] = y[j+j1, i+i1]

    mask = np.zeros([nlat_new, nlon_new], dtype=np.float32)

    zlevel = np.zeros(nlevel, dtype=np.float32)
    for n in range(nlevel):
        zlevel[n] = file.read('float')
    nlevel_new = nlevel
    ts1 = 0
    firstdate = fromyear + (frommonth-1)/12  #start at the beginning of a given month
    lastdate = toyear + tomonth/12  ## stop at the end of a given month
    ts1 = get_tcount_start(zlevel, nlevel, firstdate)
    ts2 = get_tcount_end(zlevel, nlevel, lastdate)
    nlevel_new = ts2-ts1+1

    zlevel_new = np.zeros(nlevel_new, dtype=np.float32)
    for n in range(nlevel_new):
        zlevel_new[n] = zlevel[n+ts1]

    for j in range(nlat):
        for i in range(nlon):
            temp = file.read('int32')
            if i2 > i >= i1 and j2 > j >= j1:
                mask[j-j1, i-i1] = temp

    data = np.zeros([nlevel_new, nlat_new, nlon_new], dtype=np.float32)
    t_count = ts1
    if t_count < 0:
        t_count = 0

    print("Start reading data time series skipping %s and reading for %s time steps" % (t_count, nlevel_new))

    nbytetoskip = nlon*nlat*t_count * 4
    file.move(nbytetoskip)

    val = 0
    for n in range(nlevel_new):
        for j in range(nlat)[::-1]:
            for i in range(nlon):
                val = file.read('float')
                if i2 > i >= i1 and j2 > j >= j1:
                    data[n, j-j1, i-i1] = val*1852/(30*24*60*60)  #Convert from nm/m to m/s

    file.close()


    # Create datetime objects for t index
    times = [0] * nlevel_new
    dt = np.round((enddate-startdate)/nlevel*365)
    for t in range(nlevel_new):
        times[t] = datetime(int(fromyear + np.floor(t/12)), (t%12)+1, 15, 0, 0, 0)
    origin = datetime(1970, 1, 1, 0, 0, 0)
    times_in_s = times
    for t in range(len(times)):
        times_in_s[t] = (times[t] - origin).total_seconds()
    times_in_s = np.array(times_in_s, dtype=np.float32)

    return Field(name, data, lon=x[0,:], lat=y[:,0][::-1], time=times_in_s, time_origin=origin)


def Create_TaggedFish_Class(type=JITParticle):
    class TaggedFish(type):
        monthly_age = Variable("monthly_age", dtype=np.int32)
        age = Variable('age', to_write=False)
        Vmax = Variable('Vmax', to_write=False)
        Dv_max = Variable('Dv_max', to_write=False)
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
        release_time = Variable('release_time', dtype=np.float32, initial=0, to_write=False)

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(TaggedFish, self).__init__(*args, **kwargs)
            self.setAge(4.)
            self.H = self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0
            self.taxis_scale = 0
            self.active = 0
            self.release_time = 0

        def setAge(self, months):
            self.age = months*30*24*60*60
            self.monthly_age = int(self.age/(30*24*60*60))
            self.Vmax = V_max(self.monthly_age)

    return TaggedFish
