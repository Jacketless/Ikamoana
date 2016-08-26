from parcels import *
import numpy as np
import math


def SampleH(particle, grid, time, dt):
    particle.H = grid.H[time, particle.lon, particle.lat]


def AgeParticle(particle, grid, time, dt):
    particle.age += dt
    if particle.age - (particle.monthly_age*30*24*60*60) > (30*24*60*60):
        particle.monthly_age += 1
        particle.Vmax = V_max_C(particle.monthly_age)
        particle.fish *= 1-Mortality_C(particle.monthly_age, particle.H)


def RK4(fieldx, fieldy, lon, lat, time, dt):
    f_lat = dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(lat*math.pi/180)
    u1 = fieldx[time, lon, lat]
    v1 = fieldy[time, lon, lat]
    lon1, lat1 = (lon + u1*.5*f_lon, lat + v1*.5*f_lat)
    #print('lon1 = %s, lat1 = %s' % (lon1, lat1))
    u2, v2 = (fieldx[time + .5 * dt, lon1, lat1], fieldy[time + .5 * dt, lon1, lat1])
    lon2, lat2 = (lon + u2*.5*f_lon, lat + v2*.5*f_lat)
    #print('lon2 = %s, lat2 = %s' % (lon2, lat2))
    u3, v3 = (fieldx[time + .5 * dt, lon2, lat2], fieldy[time + .5 * dt, lon2, lat2])
    lon3, lat3 = (lon + u3*f_lon, lat + v3*f_lat)
    #print('lon3 = %s, lat3 = %s' % (lon3, lat3))
    u4, v4 = (fieldx[time + dt, lon3, lat3], fieldy[time + dt, lon3, lat3])
    Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
    Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
    return [Vx, Vy]


def RK4alt(fieldx, fieldy, lon, lat, time, dt):
    u1 = fieldx[time, lon, lat]
    v1 = fieldy[time, lon, lat]
    lon1, lat1 = (lon + u1*.5*dt, lat + v1*.5*dt)
    #print('lon1 = %s, lat1 = %s' % (lon1, lat1))
    u2, v2 = (fieldx[time + .5 * dt, lon1, lat1], fieldy[time + .5 * dt, lon1, lat1])
    lon2, lat2 = (lon + u2*.5*dt, lat + v2*.5*dt)
    #print('lon2 = %s, lat2 = %s' % (lon2, lat2))
    u3, v3 = (fieldx[time + .5 * dt, lon2, lat2], fieldy[time + .5 * dt, lon2, lat2])
    lon3, lat3 = (lon + u3*dt, lat + v3*dt)
    #print('lon3 = %s, lat3 = %s' % (lon3, lat3))
    u4, v4 = (fieldx[time + dt, lon3, lat3], fieldy[time + dt, lon3, lat3])
    Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
    Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
    return [Vx, Vy]


def GradientRK4(particle, grid, time, dt):
    #print('Taxis')
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    V = RK4(grid.dH_dx, grid.dH_dy, particle.lon, particle.lat, time, dt)
    #V = [grid.dH_dx[time,particle.lon,particle.lat], grid.dH_dy[time,particle.lon,particle.lat]]
    particle.Vx = V[0] * particle.Vmax * dt * (1000*1.852*60 * math.cos(particle.lat*math.pi/180)) * to_lon
    particle.Vy = V[1] * particle.Vmax * dt * (1000*1.852*60) * to_lat


def LagrangianDiffusion(particle, grid, time, dt):
    #print('Diffusion')
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r = 1/3.
    #Rx = np.random.uniform(-1., 1.)
    #Ry = np.random.uniform(-1., 1.)
    Rx = rng.uniform(-1., 1.)
    Ry = rng.uniform(-1., 1.)
    #dK = RK4(grid.dK_dx, grid.dK_dy, particle.lon, particle.lat, time, dt)
    dK = [grid.dK_dx[time, particle.lon, particle.lat], grid.dK_dy[time, particle.lon, particle.lat]]
    half_dx = 0.5 * dK[0] * dt * to_lon
    half_dy = 0.5 * dK[1] * dt * to_lat
    #print(particle.lon + half_dx)
    #print(particle.lat + half_dy)
    #K = RK4(grid.K, grid.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)
    K = [grid.K[time, particle.lon, particle.lat]] * 2
    Rx_component = Rx * np.sqrt(2 * K[0] * dt / r) * to_lon
    Ry_component = Ry * np.sqrt(2 * K[1] * dt / r) * to_lat
    CorrectionX = dK[0] * dt * to_lon
    CorrectionY = dK[1] * dt * to_lat
    #print(Rx_component)
    #print(Ry_component)
    particle.Dx = Rx_component
    particle.Dy = Ry_component
    particle.Cx = CorrectionX
    particle.Cy = CorrectionY


def Advection(particle, grid, time, dt):
    #print('Advection')
    #to_lat = 1 / 1000. / 1.852 / 60.
    #to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    #physical_forcing = RK4(grid.U, grid.V, particle.lon, particle.lat, time, dt)
    #particle.Ax = physical_forcing[0] * dt * to_lon
    #particle.Ay = physical_forcing[1] * dt * to_lat
    physical_forcing = RK4alt(grid.U, grid.V, particle.lon, particle.lat, time, dt)
    #physical_forcing = [grid.U[time, particle.lon, particle.lat], grid.V[time, particle.lon, particle.lat]]
    particle.Ax = physical_forcing[0] * dt
    particle.Ay = physical_forcing[1] * dt


def RandomWalkDiffusion(particle, grid, time, dt):
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    dK = RK4(grid.dK_dx, grid.dK_dy, particle.lon, particle.lat, time, dt)
    half_dx = 0.5 * dK[0] * to_lon * dt
    half_dy = 0.5 * dK[1] * to_lat * dt
    Rand = np.random.uniform(0, 1.)
    K_at_half = RK4(grid.K, grid.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)[0]
    R = Rand * K_at_half * dt #np.sqrt(4 * K_at_half * dt)
    angle = np.random.uniform(0, 2*np.pi)
    CorrectionX = dK[0] * dt * to_lon
    CorrectionY = dK[1] * dt * to_lat
    particle.Cx = CorrectionX
    particle.Cy = CorrectionY
    particle.Dx = R*np.cos(angle) * to_lon
    particle.Dy = R*np.sin(angle) * to_lat


# Kernel to call a generic particle update function
def Update(particle, grid, time, dt):
    particle.update()


def Move(particle, grid, time, dt):
    # print("Ax = %s, Vs = %s, Dx = %s, Cx = %s" % (particle.Ax, particle.Vx, particle.Dx, particle.Cx))
    newlon = particle.lon + particle.Ax + particle.Dx + particle.Cx + particle.Vx
    newlat = particle.lat + particle.Ay + particle.Dy + particle.Cy + particle.Vy
    inside_bounds = False
    while inside_bounds is False:
         try:
             grid.U[time, newlon, newlat]
         except (IndexError, ValueError):
             print("Redrawing Diffusion p at %s - %s" % (newlon, newlat))
             print("inside_bounds = %s" % inside_bounds)
             LagrangianDiffusion(particle, grid, time, dt)
             newlon = particle.lon + particle.Ax + particle.Dx + particle.Cx + particle.Vx
             newlat = particle.lat + particle.Ay + particle.Dy + particle.Cy + particle.Vy
         else:
             inside_bounds = True

    particle.lon += particle.Ax + particle.Dx + particle.Cx + particle.Vx
    particle.lat += particle.Ay + particle.Dy + particle.Cy + particle.Vy
