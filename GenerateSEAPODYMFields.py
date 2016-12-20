from parcels import *
from SEAPODYM_functions import *
from netCDF4 import Dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    forcingU = 'SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc'
    forcingV = 'SEAPODYM_Forcing_Data/SEAPODYM2003_PHYS_Prepped.nc'
    forcingH = 'SEAPODYM_Forcing_Data/2002_Fields/HABITAT/2003_HABITAT.nc'

    filenames = {'U': forcingU, 'V': forcingV, 'H': forcingH}
    variables = {'U': 'u', 'V': 'v', 'H': 'habitat'}
    dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}

    print("Creating Grid")
    grid = Grid.from_netcdf(filenames=filenames, variables=variables, dimensions=dimensions, vmax=200)
    #print("Creating Diffusion Field")
    #K = Create_SEAPODYM_Diffusion_Field(grid.H, 30*24*60*60)
    #grid.add_field(K)

    print("Creating Diffusion Field with SEAPODYM units")
    S_K = Create_SEAPODYM_Diffusion_Field(grid.H, 30*24*60*60, units='nm_per_mon', P=3)
    S_K.name = 'S_K'
    grid.add_field(S_K)
    grid.write('SEAPODYM_Grid_Comparison')

    print("Loading Diffusion Field")
    bfile = Dataset('DIFFUSION_last_cohort/last_cohort_diffusion.nc', 'r')
    bX = bfile.variables['longitude']
    bY = bfile.variables['latitude']
    bT = bfile.variables['time']
    bVar = bfile.variables['skipjack_diffusion_rate']
    hfile = Dataset('SEAPODYM_Forcing_Data/SEAPODYM2003_HABITAT_Prepped.nc')
    hX = hfile.variables['lon']
    hY = hfile.variables['lat']
    hT = hfile.variables['time']
    hVar = hfile.variables['habitat']
    #I_K = Field.from_netcdf('I_K', {'lon':'longitude', 'lat':'latitude', 'time':'time'},
    #                        ['DIFFUSION/INTERIM-NEMO-PISCES_skipjack_diffusion_rate_20030115.nc',
    #                         'DIFFUSION/INTERIM-NEMO-PISCES_skipjack_diffusion_rate_20030215.nc'])
    I_K = Field('I_K', data=bVar, lon=bX, lat=bY, time=bT, transpose=True)
    grid.add_field(I_K)

    comparison = np.zeros(np.shape(I_K.data), dtype=np.float32)

    loaded = np.array(bVar)

    comparison = abs(S_K.data-loaded)
    print(np.min(comparison))
    print(np.max(comparison))


    fig = plt.figure(1)
    ax = plt.axes()
    plt.contourf(bX[:], bY[:], bVar[59, :, :], vmin=0, vmax=80000)
    plt.title("Inna's K (final timestep)")
    plt.colorbar()
    fig = plt.figure(2)
    ax = plt.axes()
    plt.contourf(bX[:], bY[:], S_K.data[59, :, :], vmin=0, vmax=80000)
    plt.title("Calculated K (final timestep)")
    plt.colorbar()
    fig = plt.figure(3)
    ax = plt.axes()
    plt.contourf(bX[:], bY[:], comparison[59, :, :], vmax=10000)
    plt.show()

    print(np.shape(hVar))
    print(np.shape(bVar))

    #example habitat to diffusion calculations
    x = [147, 110, 68, 120]
    y = [85, 64, 44, 60]
    for i in range(len(x)):
        print("Final timestep H at %s-%s = %s" % (grid.H.lon[x[i]], grid.H.lat[y[i]], grid.H.data[59, y[i], x[i]]))
        print("Final timestep K at %s-%s = %s" % (grid.S_K.lon[x[i]], grid.S_K.lat[y[i]], grid.S_K.data[59, y[i], x[i]]))
        print("Final timestep H file at %s-%s = %s" % (hX[x[i]], hY[y[i]], hVar[59, :, y[i], x[i]]))
        print("Final timestep Inna K at %s-%s = %s" % (bX[x[i]], bY[y[i]], bVar[59, y[i], x[i]]))


    #grid.write('FieldTest')