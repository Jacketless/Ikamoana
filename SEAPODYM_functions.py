import numpy as np
import struct
import math
from parcels.field import Field, Geographic, GeographicPolar
from parcels.fieldset import FieldSet,  data_converters_func
from parcels.particle import *
from netCDF4 import num2date
from datetime import datetime
from py import path
from progressbar import ProgressBar
from glob import glob


def getGradient(field, landmask=None, shallow_sea_zero=True):
    dx, dy = field.cell_distances()
    data = field.data
    dVdx = np.zeros(data.shape, dtype=np.float32)
    dVdy = np.zeros(data.shape, dtype=np.float32)
    if landmask is not None:
        landmask = np.transpose(landmask.data[0,:,:])
        if shallow_sea_zero is False:
            landmask[np.where(landmask == 2)] = 0
        for t in range(len(field.time)):
            for x in range(1, len(field.lon)-1):
                for y in range(1, len(field.lat)-1):
                    if landmask[x, y] < 1:
                        if landmask[x+1, y] == 1:
                            dVdx[t,y,x] = (data[t,y,x] - data[t,y,x-1])/dx[y, x]
                        elif landmask[x-1, y] == 1:
                            dVdx[t,y,x] = (data[t,y,x+1] - data[t,y,x])/dx[y, x]
                        else:
                            dVdx[t,y,x] = (data[t,y,x+1] - data[t,y,x-1])/(2*dx[y, x])
                        if landmask[x, y+1] == 1:
                            dVdy[t,y,x] = (data[t,y,x] - data[t,y-1,x])/dy[y, x]
                        elif landmask[x, y-1] == 1:
                            dVdy[t,y,x] = (data[t,y+1,x] - data[t,y,x])/dy[y, x]
                        else:
                            dVdy[t,y,x] = (data[t,y+1,x] - data[t,y-1,x])/(2*dy[y, x])
            # Edges always forward or backwards differencing
            for x in range(len(field.lon)):
                dVdy[t, 0, x] = (data[t, 1, x] - data[t, 0, x]) / dy[0, x]
                dVdy[t, len(field.lat)-1, x] = (data[t, len(field.lat)-1, x] - data[t, len(field.lat)-2, x]) / dy[len(field.lat)-2, x]
            for y in range(len(field.lat)):
                dVdx[t, y, 0] = (data[t, y, 1] - data[t, y, 0]) / dx[y, x]
                dVdx[t, y, len(field.lon)-1] = (data[t, y, len(field.lon)-1] - data[t, y, len(field.lon)-2]) / dx[y, x]

    return Field('d' + field.name + '_dx', dVdx, field.lon, field.lat, field.depth, field.time, \
                 interp_method=field.interp_method, allow_time_extrapolation=field.allow_time_extrapolation),\
           Field('d' + field.name + '_dy', dVdy, field.lon, field.lat, field.depth, field.time, \
                 interp_method=field.interp_method, allow_time_extrapolation=field.allow_time_extrapolation)


def V_max(monthly_age, a=2.225841100458143, b=0.8348850216641774):
    L = GetLengthFromAge(monthly_age)
    V = a * np.power(L, b)
    return V


def V_max_C(monthly_age):
    a=2.225841100458143# 0.7343607395421234 old parameters
    b=0.8348850216641774# 0.5006692114850767 old parameters
    L = GetLengthFromAge(monthly_age)
    V = a * math.pow(L, b)
    return V


def GetLengthFromAge(monthly_age):
    # Linf = 88.317
    # K = 0.1965
    # return Linf * (1 - np.exp(-K * age))
    # Hard code discrete age-lengths for now
    lengths = [3.00, 4.51, 6.02, 11.65, 16.91, 21.83, 26.43, 30.72, 34.73, 38.49, 41.99, 45.27,
               48.33, 51.19, 53.86, 56.36, 58.70, 60.88, 62.92, 64.83, 66.61, 68.27, 69.83, 71.28,
               72.64, 73.91, 75.10, 76.21, 77.25, 78.22, 79.12, 79.97, 80.76, 81.50, 82.19, 82.83,
               83.44, 84.00, 84.53, 85.02, 85.48, 85.91, 86.31, 86.69, 87.04, 87.37, 87.68, 87.96,
               88.23, 88.48, 88.71, 88.93, 89.14, 89.33, 89.51, 89.67, 89.83, 89.97, 90.11, 90.24,
               90.36, 90.47, 90.57, 90.67, 91.16]
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


def getPopFromDensityField(grid, density_field='Start'):
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
    total_fish = np.sum(getattr(grid, density_field).data * area)
    return total_fish


def Create_Landmask(grid, lim=1e-26):
    def isocean(p, lim):
        return 1 if p < lim else 0
    def isshallow(p, lim):
        return 1 if p < lim else 0

    nx = grid.H.lon.size
    ny = grid.H.lat.size
    mask = np.zeros([nx, ny, 1], dtype=np.int8)
    pbar = ProgressBar()
    for i in pbar(range(nx)):
        for j in range(1, ny-1):
            if isshallow(np.abs(grid.bathy.data[0, 2, j, i]), lim):
                mask[i,j] = 2
            if isocean(grid.H.data[0, j, i],lim):  # For each land point
                mask[i,j] = 1

    Mask = Field('LandMask', mask, grid.H.lon, grid.H.lat, transpose=True)
    Mask.interp_method = 'nearest'
    return Mask#ClosestLon, ClosestLat


def Create_SEAPODYM_F_Field(E, start_age=4, q=0.001032652877899101, selectivity_func=3, mu= 52.56103941719986, sigma=8.614813906820441, r_asymp=0.2456242856428466):
    F_Data = np.zeros(np.shape(E.data), dtype=np.float32)
    age = start_age
    for t in range(E.time.size):
        if E.time[t] - E.time[0] > (age-start_age+1)*28*24*60*60:
            age += 1
        l = GetLengthFromAge(age)*100
        if l > mu:
            Selectivity = r_asymp+(1-r_asymp)*np.exp(-(pow(l-mu,2)/(sigma)))
        else:
            Selectivity = np.exp(-(pow(l-mu,2)/(sigma)))

        print("Age = %s months, Length = %scm, Selectivity = %s" % (age, l, Selectivity))
        F_Data[t,:,:] = E.data[t,:,:] * q * Selectivity
    return(Field('F', F_Data, E.lon, E.lat, time=E.time, interp_method='nearest'))


def Create_SEAPODYM_Taxis_Fields(dHdx, dHdy, start_age=4, taxis_scale=1, units='m_per_s'):
    Tx = np.zeros(np.shape(dHdx.data), dtype=np.float32)
    Ty = np.zeros(np.shape(dHdx.data), dtype=np.float32)
    months = start_age
    age = months*30*24*60*60
    for t in range(dHdx.time.size):
        age = (start_age+t)*30*24*60*60
        if age - (months*30*24*60*60) >= (30*24*60*60):
            months += 1
        if units is 'nm_per_mon':
            for x in range(dHdx.lon.size):
                for y in range(dHdx.lat.size):
                    Tx[t, y, x] = V_max(months)*((30*24*60*60)/1852) * dHdx.data[t, y, x] * taxis_scale * (1000*1.852*60 * math.cos(dHdx.lat[y]*math.pi/180)) * 1/(60*60*24*30)
                    Ty[t, y, x] = V_max(months)*((30*24*60*60)/1852) * dHdy.data[t, y, x] * taxis_scale * (1000*1.852*60) * 1/(60*60*24*30)
        else:
            for x in range(dHdx.lon.size):
                for y in range(dHdx.lat.size):
                    Tx[t, y, x] = V_max(months) * dHdx.data[t, y, x] * taxis_scale * (1000*1.852*60 * math.cos(dHdx.lat[y]*math.pi/180))# / ((1 / 1000. / 1.852 / 60.) / math.cos(dHdx.lat[y]*math.pi/180))
                    Ty[t, y, x] = V_max(months) * dHdy.data[t, y, x] * taxis_scale * (1000*1.852*60)#/ (1 / 1000. / 1.852 / 60.)

    return [Field('Tx', Tx, dHdx.lon, dHdx.lat, time=dHdx.time, interp_method='nearest', allow_time_extrapolation=True),
            Field('Ty', Ty, dHdx.lon, dHdx.lat, time=dHdx.time, interp_method='nearest', allow_time_extrapolation=True)]


def Create_SEAPODYM_Diffusion_Field(H, timestep=30*24*60*60, sigma=0.1769952864978924, c=0.662573993401526, P=3,
                                    start_age=4, Vmax_slope=1, units='m_per_s',
                                    diffusion_boost=0, diffusion_scale=1, sig_scale=1, c_scale=1,
                                    verbose=True):
    # Old parameters sigma=0.1999858740340303, c=0.9817751085550976,
    K = np.zeros(np.shape(H.data), dtype=np.float32)
    months = start_age
    age = months*30*24*60*60
    for t in range(H.time.size):
        # Increase age in months if required, to incorporate appropriate Vmax
        # months in SEAPODYM are all assumed to be 30 days long
        #age = H.time[t] - H.time[0] + start_age*30*24*60*60 # this is for 'true' ageing
        age = (start_age+t)*30*24*60*60
        if age - (months*30*24*60*60) >= (30*24*60*60):
            months += 1
        if verbose:
            print('age in days = %s' % (age/(24*60*60)))
            print("Calculating diffusivity for fish aged %s months" % months)
        if units == 'nm_per_mon':
            Dmax = (np.power(GetLengthFromAge(months)*((30*24*60*60)/1852), 2) / 4 ) * timestep/(60*60*24*30) #vmax = L for diffusion
        else:
            Dmax = (np.power(GetLengthFromAge(months), 2) / 4) * timestep  #fixed b parameter for diffusion
        sig_D = sigma * Dmax
        for x in range(H.lon.size):
            for y in range(H.lat.size):
                K[t, y, x] = sig_scale * sig_D * (1 - c_scale * c * np.power(H.data[t, y, x], P)) * diffusion_scale + diffusion_boost

    return Field('K', K, H.lon, H.lat, time=H.time, interp_method='nearest', allow_time_extrapolation=True)


def Create_SEAPODYM_Grid(forcing_files, startD=None, startD_dims=None,
                         forcing_vars={'U': 'u', 'V': 'v', 'H': 'habitat'},
                         forcing_dims={'lon': 'lon', 'lat': 'lon', 'time': 'time'}, K_timestep=30*24*60*60,
                         diffusion_file=None, field_units='m_per_s',
                         diffusion_dims={'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'data': 'skipjack_diffusion_rate'},
                         scaleH=None, start_age=4, output_density=False, diffusion_scale=1, sig_scale=1, c_scale=1,
                         verbose=False):
    if startD_dims is None:
        startD_dims = forcing_dims
    if verbose:
        print("Creating Grid\nLoading files:")
        for f in forcing_files.values():
            print(f)

    grid = FieldSet.from_netcdf(filenames=forcing_files, variables=forcing_vars, dimensions=forcing_dims,
                            vmin=-200, vmax=1e34, interp_method='nearest', allow_time_extrapolation=True)
    print(forcing_files['U'])
    Depthdata = Field.from_netcdf('u', dimensions={'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': 'depth'},
                                     filenames=forcing_files['U'], allow_time_extrapolation=True)
    Depthdata.name = 'bathy'
    grid.add_field(Depthdata)


    if startD is not None:
        grid.add_field(Field.from_netcdf('Start', dimensions=startD_dims,
                                         filenames=path.local(startD), vmax=1000,
                                         interp_method='nearest', allow_time_extrapolation=True))

    if output_density:
        # Add a density field that will hold particle densities
        grid.add_field(Field('Density', np.full([grid.U.lon.size, grid.U.lat.size, grid.U.time.size],-1, dtype=np.float64),
                       grid.U.lon, grid.U.lat, depth=grid.U.depth, time=grid.U.time, transpose=True))

    LandMask = Create_Landmask(grid)
    grid.add_field(LandMask)
    grid.U.data[grid.U.data > 1e5] = 0
    grid.V.data[grid.V.data > 1e5] = 0
    grid.H.data[grid.H.data > 1e5] = 0
    # Scale the H field between zero and one if required
    if scaleH is not None:
        grid.H.data /= np.max(grid.H.data)
        grid.H.data[np.where(grid.H.data < 0)] = 0
        grid.H.data *= scaleH

    # Offline calculate the 'diffusion' grid as a function of habitat
    if verbose:
        print("Creating Diffusion Field")
    if diffusion_file is None:
        K = Create_SEAPODYM_Diffusion_Field(grid.H, timestep=K_timestep, start_age=start_age,
                                            diffusion_scale=diffusion_scale, units=field_units,
                                            sig_scale=sig_scale, c_scale=c_scale, verbose=verbose)
    else:
        if verbose:
            print("Loading from file: %s" % diffusion_file)
        K = Field.from_netcdf('K', diffusion_dims, [diffusion_file], interp_method='nearest', vmax=1000000)
        if diffusion_units == 'nm2_per_mon':
            K.data *= 1.30427305

    grid.add_field(K)

    if verbose:
        print("Calculating H Gradient Fields")
    dHdx, dHdy = getGradient(grid.H, grid.LandMask)
    grid.add_field(dHdx)
    grid.add_field(dHdy)
    #gradients = grid.H.gradient()
    #for field in gradients:
    #    grid.add_field(field)

    if verbose:
        print("Calculating Taxis Fields")
    T_Fields = Create_SEAPODYM_Taxis_Fields(grid.dH_dx, grid.dH_dy, start_age=start_age, units=field_units)
    for field in T_Fields:
        grid.add_field(field)

    if verbose:
        print("Creating combined Taxis and Advection field")
    grid.add_field(Field('TU', grid.U.data+grid.Tx.data, grid.U.lon, grid.U.lat, time=grid.U.time, vmin=-200, vmax=1e34,
                         interp_method='nearest', allow_time_extrapolation=True))# units=unit_converters('spherical')[0]))
    grid.add_field(Field('TV', grid.V.data+grid.Ty.data, grid.U.lon, grid.U.lat, time=grid.U.time, vmin=-200, vmax=1e34,
                         interp_method='nearest', allow_time_extrapolation=True))#, units=unit_converters('spherical')[1]))


    if verbose:
        print("Calculating K Gradient Fields")
    #K_gradients = grid.K.gradient()
    #for field in K_gradients:
    #    grid.add_field(field)
    dKdx, dKdy = getGradient(grid.K, grid.LandMask, False)
    grid.add_field(dKdx)
    grid.add_field(dKdy)
    grid.K.interp_method = grid.dK_dx.interp_method = grid.dK_dy.interp_method = grid.H.interp_method = \
                           grid.dH_dx.interp_method = grid.dH_dy.interp_method = grid.U.interp_method = grid.V.interp_method = 'nearest'

    #grid.K.allow_time_extrapolation = grid.dK_dx.allow_time_extrapolation = grid.dK_dy.allow_time_extrapolation = \
    #                                  grid.H.allow_time_extrapolation = grid.dH_dx.allow_time_extrapolation = \
    #                                  grid.dH_dy.allow_time_extrapolation = grid.U.allow_time_extrapolation = \
    #                                  grid.V.allow_time_extrapolation = True

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


def getSEAPODYMarguments(run="SEAPODYM_2003"):
    args = {}
    if run == "SEAPODYM_2003":
        args.update({'ForcingFiles': {'U': 'SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month*.nc',
                                      'V': 'SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month*.nc',
                                      'H': 'SEAPODYM_Forcing_Data/Latest/HABITAT/2003Run/INTERIM-NEMO-PISCES_skipjack_habitat_index_*.nc',
                                      'start': 'SEAPODYM_Forcing_Data/Latest/DENSITY/INTERIM-NEMO-PISCES_skipjack_cohort_20021015_density_M0_20030115.nc',
                                      'E': 'SEAPODYM_Forcing_Data/Latest/FISHERIES/P3_E_Data.nc'}})
        args.update({'ForcingVariables': {'U': 'u',
                                          'V': 'v',
                                          'H': 'skipjack_habitat_index',
                                          'start': 'skipjack_cohort_20021015_density_M0',
                                          'E': 'P3'}})
        args.update({'LandMaskFile': 'SEAPODYM_Forcing_Data/Latest/PHYSICAL/2003Run/2003run_PHYS_month01.nc'})
        args.update({'Filestems': {'U': 'm',
                                   'V': 'm',
                                   'H': 'd'}})
        args.update({'Kernels': ['Advection_C',
                                 'LagrangianDiffusion',
                                 'TaxisRK4',
                                 'FishingMortality',
                                 'NaturalMortality',
                                 'MoveWithLandCheck',
                                 'AgeAnimal']})

    if run == "DiffusionTest":
        print("in")
        args.update({'ForcingFiles': {'U': 'DiffusionExample/DiffusionExampleCurrents.nc',
                                      'V': 'DiffusionExample/DiffusionExampleCurrents.nc',
                                      'H': 'DiffusionExample/DiffusionExampleHabitat.nc'}})
        args.update({'ForcingVariables': {'U': 'u',
                                          'V': 'v',
                                          'H': 'skipjack_habitat_index'}})
        args.update({'Filestems': {'U': 'm',
                                   'V': 'm',
                                   'H': 'd'}})
        args.update({'Kernels': ['LagrangianDiffusion',
                                 'Move',
                                 'MoveOffLand']})
    return args
