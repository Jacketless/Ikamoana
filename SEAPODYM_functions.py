import numpy as np
from parcels.field import Field
from parcels.grid import Grid

def V_max(monthly_age, a=0.7343607395421234, b=0.5006692114850767):
    L = GetLengthFromAge(monthly_age)
    V = a * np.power(L, b)
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
               85.48, 85.91, 86.31, 86.69, 87.04, 87.37, 87.68, 87.96, 89.42, 89.42, 89.42, 89.42,
               89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42,
               89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42,
               89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42]
    return lengths[monthly_age-1]/100 # Convert to meters


def Mortality(age, MPmax=0.3, MPexp=0.1008314958945224, MSmax=0.006109001382111822, MSslope=0.8158285706493162,
              Mrange=1.430156372206337e-05, H=1):
    Mnat = MPmax*np.exp(-MPexp*age) + MSmax*np.power(age, MSslope)
    Mvar = Mnat * np.power(1 - Mrange, 1-H/2)
    return Mvar


def Create_SEAPODYM_Diffusion_Field(H, timestep=86400, sigma=0.1999858740340303, c=0.9817751085550976, P=3,
                                    start_age=4, Vmax_slope=1):
    K = np.zeros(np.shape(H.data), dtype=np.float32)
    months = start_age
    age = months*30*24*60*60
    for t in range(H.time.size):
        print("H field time = %s" % H.time[t])
        # Increase age in months if required, to incorporate appropriate Vmax
        age = H.time[t] - H.time[0]
        if age - (months*30*24*60*60) > (30*24*60*60):
            months += 1
        print("Fish age in months = %s" % months)
        Dmax = np.power(V_max(months, b=Vmax_slope), 2) / 4 * timestep #fixed b parameter for diffusion
        sig_D = sigma * Dmax
        for x in range(H.lon.size):
            for y in range(H.lat.size):
                K[t, y, x] = sig_D * (1 - c * np.power(H.data[t, y, x], P))

    return Field('K', K, H.lon, H.lat, time=H.time)


def Create_SEAPODYM_Grid(forcingU, forcingV, forcingH, startD=None,
                         Uname='u', Vname='v', Hname='habitat', Dname='density',
                         dimLon='lon', dimLat='lat', dimTime='time',
                         scaleH=False, output_density=False):
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
    if scaleH:
        grid.H.data /= np.max(grid.H.data)
        grid.H.data[np.where(grid.H.data < 0)] = 0

    # Offline calculate the 'diffusion' grid as a function of habitat
    print("Creating Diffusion Field")
    K = Create_SEAPODYM_Diffusion_Field(grid.H, 24*60*60)
    grid.add_field(K)

    # Offline calculation of the diffusion and basic habitat grid
    print("Calculating Gradient Fields")
    gradients = grid.H.gradient()
    for field in gradients:
        grid.add_field(field)
    K_gradients = grid.K.gradient()
    for field in K_gradients:
        grid.add_field(field)

    return grid
