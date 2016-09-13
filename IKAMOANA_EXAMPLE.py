from parcels import *
from SEAPODYM_functions import *
from Behaviour import *
from py import path
from glob import glob
import numpy as np
import math
from argparse import ArgumentParser


def Drifter():

    grid = Grid.from_netcdf(filenames={'U':'/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_u_1993_01.TropPac.nc',
                                       'V':'/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_v_1993_01.TropPac.nc'},
                            variables={'U':'u', 'V':'v'},
                            dimensions={'lon':'xu_ocean', 'lat':'yu_ocean','time':'Time'},
                            vmin=-200, vmax=200)

    drifterset = grid.ParticleSet(size=5, pclass=ScipyParticle,
                                  lon=np.linspace(152, 153,5),
                                  lat=np.linspace(-8, -6, 5))

    drifterset.execute(AdvectionRK4, endtime=grid.time[0]+(60*60*24*30), dt=60*60,
                    output_file=drifterset.ParticleFile(name="drifter_results"),
                    interval=60*60)


def Diffuser():
    grid = Grid.from_netcdf(filenames={'U':'/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_u_1993_01.TropPac.nc',
                                       'V':'/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_v_1993_01.TropPac.nc'},
                            variables={'U':'u', 'V':'v'},
                            dimensions={'lon':'xu_ocean', 'lat':'yu_ocean','time':'Time'},
                            vmin=-200, vmax=200)

    grid.add_field(CreateGradientField(grid))
    K_gradients = grid.K.gradient()
    for field in K_gradients:
        grid.add_field(field)

    Diffuser = Create_Particle_Class(ScipyParticle)

    print(np.mean(grid.U.lon))
    print(np.mean(grid.U.lat))

    diffuserset = grid.ParticleSet(size=5, pclass=Diffuser,
                                   lon=[np.mean(grid.U.lon)+0.5]*5,
                                   lat=[np.mean(grid.U.lat)+0.5]*5)

    diffuserset.execute(diffuserset.Kernel(LagrangianDiffusion) + diffuserset.Kernel(Move), endtime=grid.time[0]+(60*60*24*30), dt=60*60,
                    output_file=diffuserset.ParticleFile(name="diffuser_results"),
                    interval=60*60)
    grid.write('diffusergrid')

def Climber():
    grid = Grid.from_netcdf(filenames={'U':'/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_u_1993_01.TropPac.nc',
                                       'V':'/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1/ocean_v_1993_01.TropPac.nc'},
                            variables={'U':'u', 'V':'v'},
                            dimensions={'lon':'xu_ocean', 'lat':'yu_ocean','time':'Time'},
                            vmin=-200, vmax=200)

    grid.add_field(CreateGradientField(grid, name='H'))
    H_gradients = grid.H.gradient()
    for field in H_gradients:
        grid.add_field(field)

    Climber = Create_Particle_Class(ScipyParticle)

    climberset = grid.ParticleSet(size=5, pclass=Climber,
                                  lon=[np.mean(grid.U.lon)+1, np.mean(grid.U.lon)-1, np.mean(grid.U.lon), np.mean(grid.U.lon), np.mean(grid.U.lon)-2],
                                  lat=[np.mean(grid.U.lat), np.mean(grid.U.lat), np.mean(grid.U.lat)+1, np.mean(grid.U.lat)-1, np.mean(grid.U.lat)+2])

    climberset.execute(climberset.Kernel(GradientRK4_C) + climberset.Kernel(Move), endtime=grid.time[0]+(60*60*24*30), dt=60*60,
                    output_file=climberset.ParticleFile(name="climber_results"),
                    interval=60*60)


def CreateGradientField(grid, name='K',mu=None):
    """Generate a simple gradient field containing a single maxima and gaussian gradient
    """
    depth = np.zeros(1, dtype=np.float32)
    time = grid.time

    lon, lat = np.arange(grid.U.lon[0], grid.U.lon[-1], 10), np.arange(grid.U.lat[0], grid.U.lat[-1], 10)

    K = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    if mu is None:
        mu = [np.mean(grid.U.lon), np.mean(grid.U.lat)]

    def MVNorm(x, y, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
        mu_x = mu[0]
        mu_y = mu[1]
        sigma = np.array(sigma)
        sig_x = sigma[0, 0]
        sig_y = sigma[1, 1]
        sig_xy = sigma[1, 0]

        pd = 1/(2 * np.pi * sig_x * sig_y * np.sqrt(1 - sig_xy))
        pd = pd * np.exp(-1 / (2 * (1 - np.power(sig_xy, 2))) * (
                        ((np.power(x - mu_x, 2)) / np.power(sig_x, 2)) +
                        ((np.power(y - mu_y, 2)) / np.power(sig_y, 2)) -
                        ((2 * sig_xy * (x - mu_x) * (y - mu_y)) / (sig_x * sig_y))))

        return pd

    sig = [[0.3, 0], [0, 0.3]]
    for i, x in enumerate(lon):
        for j, y in enumerate(lat):
            K[i, j, :] = MVNorm(x, y, mu, sig)

    # Boost to provide enough force for our gradient climbers
    K_Field = Field(name, K, lon, lat, depth, time, transpose=True)

    return K_Field


def Create_Particle_Class(type=JITParticle):

    class SimpleFish(type):
        Vmax = Variable('Vmax')

        def __init__(self, *args, **kwargs):
            """Custom initialisation function which calls the base
            initialisation and adds the instance variable p"""
            super(SimpleFish, self).__init__(*args, **kwargs)
            self.Vmax=0.1
            self.Dx = self.Dy = self.Cx = self.Cy = self.Vx = self.Vy = self.Ax = self.Ay = 0
            self.taxis_dampen = 1000

    return SimpleFish


if __name__ == "__main__":
    p = ArgumentParser(description="""
    Example of underlying habitat field""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('-t', '--type', default='drifter',
                   help='The example type (drifter, diffuser or climber)')

    args = p.parse_args()

    if args.type == 'drifter':
        print("go!")
        Drifter()
    if args.type == 'diffuser':
        Diffuser()
    if args.type == 'climber':
        Climber()
