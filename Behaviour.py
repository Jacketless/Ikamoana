from parcels import *
import numpy as np
import math


def SampleH(particle, fieldset, time, dt):
    particle.H = fieldset.H[time, particle.lon, particle.lat, particle.depth]
    particle.dHdx = fieldset.dH_dx[time, particle.lon, particle.lat, particle.depth]
    particle.dHdy = fieldset.dH_dy[time, particle.lon, particle.lat, particle.depth]


def CheckRelease(particle, fieldset, time, dt):
    if time > particle.release_time:
        particle.active = 1


def FishingMortality(particle, fieldset, time, dt):
    Fmor = (1-(fieldset.F[time, particle.lon, particle.lat, particle.depth] * (dt/(30*24*60*60))))
    particle.school = particle.school * Fmor
    particle.depletionF = Fmor


def NaturalMortality(particle, fieldset, time, dt):
    MPmax=0.3
    MPexp=0.1008314958945224
    MSmax=0.006109001382111822
    MSslope=0.8158285706493162
    Mrange=0.00001430156
    Mnat = MPmax*math.exp(-MPexp*particle.monthly_age) + MSmax*math.pow(particle.monthly_age, MSslope)
    Mvar = Mnat * math.pow(1 - Mrange, 1-fieldset.H[time, particle.lon, particle.lat, particle.depth]/2)
    Nmor = (1 - (Mvar * (dt/(30*24*60*60))))
    particle.school = particle.school * Nmor
    particle.depletionN = Nmor


def AgeAnimal(particle, fieldset, time, dt):
    particle.age += dt
    if (particle.age - (particle.monthly_age*30*24*60*60)) > (30*24*60*60):
        particle.monthly_age += 1


def AgeParticle(particle, fieldset, time, dt):
    #print("Ageing")
    particle.age += dt
    if (particle.age - (particle.monthly_age*30*24*60*60)) > (30*24*60*60):
        particle.monthly_age += 1
        # a=2.225841100458143# 0.7343607395421234 old parameters
        # b=0.8348850216641774# 0.5006692114850767 old parameters
        # lengths = [3.00, 4.51, 6.02, 11.65, 16.91, 21.83, 26.43, 30.72, 34.73, 38.49, 41.99, 45.27,
        #            48.33, 51.19, 53.86, 56.36, 58.70, 60.88, 62.92, 64.83, 66.61, 68.27, 69.83, 71.28,
        #            72.64, 73.91, 75.10, 76.21, 77.25, 78.22, 79.12, 79.97, 80.76, 81.50, 82.19, 82.83,
        #            83.44, 84.00, 84.53, 85.02, 85.48, 85.91, 86.31, 86.69, 87.04, 87.37, 87.68, 87.96,
        #            88.23, 88.48, 88.71, 88.93, 89.14, 89.33, 89.51, 89.67, 89.83, 89.97, 90.11, 90.24,
        #            90.36, 90.47, 90.57, 90.67, 91.16]
        # L = lengths[(particle.monthly_age-1)]/100  #L = GetLengthFromAge(monthly_age)
        # vmax = a * math.pow(L, b)
        # particle.Vmax = vmax
        # MPmax=0.3
        # MPexp=0.1008314958945224
        # MSmax=0.006109001382111822
        # MSslope=0.8158285706493162
        # Mrange=0.00001430156
        # Mnat = MPmax*math.exp(-1*MPexp*particle.age) + MSmax*math.pow(particle.age, MSslope)
        # Hexp = 1-fieldset.H[time, particle.lon, particle.lat, particle.depth]/2
        # Mvar = Mnat * math.pow((1 - Mrange), Hexp)#(1-fieldset.H[time, lon, lat]))#/2))
        # particle.fish *= 1-Mvar  #particle.fish *= 1-Mortality_C(particle.monthly_age, particle.H)


def AgeIndividual(particle, fieldset, time, dt):
    #print("Ageing")
    particle.age += dt
    if (particle.age - (particle.monthly_age*30*24*60*60)) > (30*24*60*60):
        particle.monthly_age += 1
        a=2.225841100458143# 0.7343607395421234 old parameters
        b=0.8348850216641774# 0.5006692114850767 old parameters
        Alengths = [3.00, 4.51, 6.02, 11.65, 16.91, 21.83, 26.43, 30.72, 34.73, 38.49,
                   41.99, 45.27, 48.33, 51.19, 53.86, 56.36, 58.70, 60.88, 62.92, 64.83,
                   66.61, 68.27, 69.83, 71.28, 72.64, 73.91, 75.10, 76.21, 77.25, 78.22,
                   79.12, 79.97, 80.76, 81.50, 82.19, 82.83, 83.44, 84.00, 84.53, 85.02,
                   85.48, 85.91, 86.31, 86.69, 87.04, 87.37, 87.68, 87.96, 89.42, 89.42, 89.42, 89.42,
                   89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42,
                   89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42,
                   89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42, 89.42]
        L = lengths[(particle.monthly_age-1)]/100  #L = GetLengthFromAge(monthly_age)
        vmax = a * math.pow(L, b)
        particle.Vmax = vmax
        MPmax=0.3
        MPexp=0.1008314958945224
        MSmax=0.006109001382111822
        MSslope=0.8158285706493162
        Mrange=0.00001430156
        Mnat = MPmax*math.exp(-1*MPexp*particle.age) + MSmax*math.pow(particle.age, MSslope)
        Hexp = 1-fieldset.H[time, particle.lon, particle.lat, particle.depth]/2
        Mvar = Mnat * math.pow((1 - Mrange), Hexp)#(1-fieldset.H[time, lon, lat, particle.depth]))#/2))
        if random.uniform(0, 1) > Mvar:
            particle.active = 0
            particle.release_time = 100000000*100000000 #Particle will not be reactivated
            particle.lon = 0
            particle.lat = 0


def RK4(fieldx, fieldy, lon, lat, time, dt):
    f_lat = dt / 1000. / 1.852 / 60.
    f_lon = f_lat / math.cos(lat*math.pi/180)
    u1 = fieldx[time, lon, lat, particle.depth]
    v1 = fieldy[time, lon, lat, particle.depth]
    lon1, lat1 = (lon + u1*.5*f_lon, lat + v1*.5*f_lat)
    #print('lon1 = %s, lat1 = %s' % (lon1, lat1))
    u2, v2 = (fieldx[time + .5 * dt, lon1, lat1, particle.depth], fieldy[time + .5 * dt, lon1, lat1, particle.depth])
    lon2, lat2 = (lon + u2*.5*f_lon, lat + v2*.5*f_lat)
    #print('lon2 = %s, lat2 = %s' % (lon2, lat2))
    u3, v3 = (fieldx[time + .5 * dt, lon2, lat2, particle.depth], fieldy[time + .5 * dt, lon2, lat2, particle.depth])
    lon3, lat3 = (lon + u3*f_lon, lat + v3*f_lat)
    #print('lon3 = %s, lat3 = %s' % (lon3, lat3))
    u4, v4 = (fieldx[time + dt, lon3, lat3, particle.depth], fieldy[time + dt, lon3, lat3, particle.depth])
    Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
    Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
    return [Vx, Vy]


def RK4alt(fieldx, fieldy, lon, lat, time, dt):
    u1 = fieldx[time, lon, lat, particle.depth]
    v1 = fieldy[time, lon, lat, particle.depth]
    lon1, lat1 = (lon + u1*.5*dt, lat + v1*.5*dt)
    #print('lon1 = %s, lat1 = %s' % (lon1, lat1))
    u2, v2 = (fieldx[time + .5 * dt, lon1, lat1, particle.depth], fieldy[time + .5 * dt, lon1, lat1, particle.depth])
    lon2, lat2 = (lon + u2*.5*dt, lat + v2*.5*dt)
    #print('lon2 = %s, lat2 = %s' % (lon2, lat2))
    u3, v3 = (fieldx[time + .5 * dt, lon2, lat2, particle.depth], fieldy[time + .5 * dt, lon2, lat2, particle.depth])
    lon3, lat3 = (lon + u3*dt, lat + v3*dt)
    #print('lon3 = %s, lat3 = %s' % (lon3, lat3))
    u4, v4 = (fieldx[time + dt, lon3, lat3, particle.depth], fieldy[time + dt, lon3, lat3, particle.depth])
    Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
    Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
    return [Vx, Vy]


def GradientRK4(particle, fieldset, time, dt):
    #print('Taxis')
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    V = RK4(fieldset.dH_dx, fieldset.dH_dy, particle.lon, particle.lat, time, dt)
    #V = [fieldset.dH_dx[time,particle.lon,particle.lat, particle.depth], fieldset.dH_dy[time,particle.lon,particle.lat, particle.depth]]
    particle.Vx = V[0] * particle.Vmax * dt * (1000*1.852*60 * math.cos(particle.lat*math.pi/180)) * to_lon
    particle.Vy = V[1] * particle.Vmax * dt * (1000*1.852*60) * to_lat


def GradientRK4_C(particle, fieldset, time, dt):
    if particle.active == 1:
        f_lat = dt / 1000. / 1.852 / 60.
        f_lon = f_lat / math.cos(particle.lat*math.pi/180)
        ## Something about the below RK4 of dH_dx that C doesn't like (referencing odd field values??)
        # u1 = fieldset.dH_dx[time, particle.lon, particle.lat, particle.depth]
        # v1 = fieldset.dH_dy[time, particle.lon, particle.lat, particle.depth]
        # lon1, lat1 = (particle.lon + u1*.5*f_lon, particle.lat + v1*.5*f_lat)
        # #print('lon1 = %s, lat1 = %s' % (lon1, lat1))
        # u2, v2 = (fieldset.dH_dx[time + .5 * dt, lon1, lat1, particle.depth], fieldset.dH_dy[time + .5 * dt, lon1, lat1, particle.depth])
        # lon2, lat2 = (particle.lon + u2*.5*f_lon, particle.lat + v2*.5*f_lat)
        # #print('lon2 = %s, lat2 = %s' % (lon2, lat2))
        # u3, v3 = (fieldset.dH_dx[time + .5 * dt, lon2, lat2, particle.depth], fieldset.dH_dy[time + .5 * dt, lon2, lat2, particle.depth])
        # lon3, lat3 = (particle.lon + u3*f_lon, particle.lat + v3*f_lat)
        # #print('lon3 = %s, lat3 = %s' % (lon3, lat3))
        # u4, v4 = (fieldset.dH_dx[time + dt, lon3, lat3, particle.depth], fieldset.dH_dy[time + dt, lon3, lat3, particle.depth])
        # Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
        # Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
        particle.Vx = fieldset.dH_dx[time, particle.lon, particle.lat, particle.depth] * particle.Vmax * (1000*1.852*60 * math.cos(particle.lat*math.pi/180)) * particle.taxis_scale * f_lon
        particle.Vy = fieldset.dH_dy[time, particle.lon, particle.lat, particle.depth] * particle.Vmax * (1000*1.852*60) * particle.taxis_scale * f_lat
        #particle.Vx = Vx #* particle.Vmax #* (1000*1.852*60 * math.cos(particle.lat*math.pi/180)) * f_lon
        #particle.Vy = Vy #* particle.Vmax * (1000*1.852*60) * f_lat


def TaxisRK4(particle, fieldset, time, dt):
    if particle.active == 1:
        f_lat = dt / 1000. / 1.852 / 60.
        f_lon = f_lat / math.cos(particle.lat*math.pi/180)
        u1 = fieldset.Tx[time, particle.lon, particle.lat, particle.depth]
        v1 = fieldset.Ty[time, particle.lon, particle.lat, particle.depth]
        lon1, lat1 = (particle.lon + u1*.5*f_lon, particle.lat + v1*.5*f_lat)
        u2, v2 = (fieldset.Tx[time + .5 * dt, lon1, lat1, particle.depth], fieldset.Ty[time + .5 * dt, lon1, lat1, particle.depth])
        lon2, lat2 = (particle.lon + u2*.5*f_lon, particle.lat + v2*.5*f_lat)
        u3, v3 = (fieldset.Tx[time + .5 * dt, lon2, lat2, particle.depth], fieldset.Ty[time + .5 * dt, lon2, lat2, particle.depth])
        lon3, lat3 = (particle.lon + u3*f_lon, particle.lat + v3*f_lat)
        u4, v4 = (fieldset.Tx[time + dt, lon3, lat3, particle.depth], fieldset.Ty[time + dt, lon3, lat3, particle.depth])
        Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
        Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
        particle.Vx = Vx * f_lat
        particle.Vy = Vy * f_lon


def CurrentAndTaxisRK4(particle, fieldset, time, dt):
    if particle.active == 1:
        u1 = fieldset.TU[time, particle.lon, particle.lat, particle.depth]
        v1 = fieldset.TV[time, particle.lon, particle.lat, particle.depth]
        lon1, lat1 = (particle.lon + u1*.5*dt, particle.lat + v1*.5*dt)
        u2, v2 = (fieldset.TU[time + .5 * dt, lon1, lat1, particle.depth], fieldset.TV[time + .5 * dt, lon1, lat1, particle.depth])
        lon2, lat2 = (particle.lon + u2*.5*dt, particle.lat + v2*.5*dt)
        u3, v3 = (fieldset.TU[time + .5 * dt, lon2, lat2, particle.depth], fieldset.TV[time + .5 * dt, lon2, lat2, particle.depth])
        lon3, lat3 = (particle.lon + u3*dt, particle.lat + v3*dt)
        u4, v4 = (fieldset.TU[time + dt, lon3, lat3, particle.depth], fieldset.TV[time + dt, lon3, lat3, particle.depth])
        Vx = (u1 + 2*u2 + 2*u3 + u4) / 6.
        Vy = (v1 + 2*v2 + 2*v3 + v4) / 6.
        particle.Ax = Vx * dt
        particle.Ay = Vy * dt
        particle.Vx = 0
        particle.Vy = 0


def FishDensityClimber(particle, fieldset, time, dt):
    if particle.active == 1:
        f_lat3 = dt / 1000. / 1.852 / 60.
        f_lon3 = f_lat3 / math.cos(particle.lat*math.pi/180)
        particle.Vx = fieldset.dFishDensity_dx[time, particle.lon, particle.lat, particle.depth] * \
                      particle.Vmax * (1000*1.852*60 * math.cos(particle.lat*math.pi/180)) * \
                      particle.taxis_scale * f_lon3 #* fieldset.MaxDensity
        particle.Vy = fieldset.dFishDensity_dy[time, particle.lon, particle.lat, particle.depth] * \
                      particle.Vmax * (1000*1.852*60) * \
                      particle.taxis_scale * f_lat3 #* fieldset.MaxDensity
        print("Vx= %s  Vy= %s" % (particle.Vx, particle.Vy))


def LagrangianDiffusion(particle, fieldset, time, dt):
    if particle.active == 1:
        to_lat = 1 / 1000. / 1.852 / 60.
        to_lon = to_lat / math.cos(particle.lat*math.pi/180)
        r_var = 1/3.
        #Rx = np.random.uniform(-1., 1.)
        #Ry = np.random.uniform(-1., 1.)
        Rx = random.uniform(-1., 1.)
        Ry = random.uniform(-1., 1.)
        #dK  = RK4(fieldset.dK_dx, fieldset.dK_dy, particle.lon, particle.lat, time, dt)
        dKdx, dKdy = (fieldset.dK_dx[time, particle.lon, particle.lat, particle.depth], fieldset.dK_dy[time, particle.lon, particle.lat, particle.depth])
        #half_dx = 0.5 * dKdx * dt * to_lon
        #half_dy = 0.5 * dKdy * dt * to_lat
        #print(particle.lon + half_dx)
        #print(particle.lat + half_dy)
        #K = RK4(fieldset.K, fieldset.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)
        Kfield = fieldset.K[time, particle.lon, particle.lat, particle.depth]
        Rx_component = Rx * math.sqrt(2 * Kfield * dt / r_var) * to_lon
        Ry_component = Ry * math.sqrt(2 * Kfield * dt / r_var) * to_lat
        CorrectionX = dKdx * dt * to_lon
        CorrectionY = dKdy * dt * to_lat
        #print(Rx_component)
        #print(Ry_component)
        particle.Dx = Rx_component
        particle.Dy = Ry_component
        particle.Cx = CorrectionX
        particle.Cy = CorrectionY
        #Dx = Rx_component
        #Dy = Ry_component
        #Cx = CorrectionX
        #Cy = CorrectionY


def Advection(particle, fieldset, time, dt):
    if particle.active == 1:
        physical_forcing = RK4alt(fieldset.U, fieldset.V, particle.lon, particle.lat, time, dt)
        particle.Ax = physical_forcing[0] * dt
        particle.Ay = physical_forcing[1] * dt


def Advection_C(particle, fieldset, time, dt):
    #print("Advection")
    if particle.active == 1:
        #to_lat = 1 / 1000. / 1.852 / 60.
        #to_lon = to_lat / math.cos(particle.lat*math.pi/180)
        u1 = fieldset.U[time, particle.lon, particle.lat, particle.depth]
        v1 = fieldset.V[time, particle.lon, particle.lat, particle.depth]
        lon1, lat1 = (particle.lon + u1*.5*dt, particle.lat + v1*.5*dt)
        #print('lon1 = %s, lat1 = %s' % (lon1, lat1))
        u2, v2 = (fieldset.U[time + .5 * dt, lon1, lat1, particle.depth], fieldset.V[time + .5 * dt, lon1, lat1, particle.depth])
        lon2, lat2 = (particle.lon + u2*.5*dt, particle.lat + v2*.5*dt)
        #print('lon2 = %s, lat2 = %s' % (lon2, lat2))
        u3, v3 = (fieldset.U[time + .5 * dt, lon2, lat2, particle.depth], fieldset.V[time + .5 * dt, lon2, lat2, particle.depth])
        lon3, lat3 = (particle.lon + u3*dt, particle.lat + v3*dt)
        #print('lon3 = %s, lat3 = %s' % (lon3, lat3))
        u4, v4 = (fieldset.U[time + dt, lon3, lat3, particle.depth], fieldset.V[time + dt, lon3, lat3, particle.depth])
        Ax = (u1 + 2*u2 + 2*u3 + u4) / 6.
        Ay = (v1 + 2*v2 + 2*v3 + v4) / 6.
        particle.Ax = Ax * dt# / to_lon #Convert back to m/s so we save to particle file in a usual format
        particle.Ay = Ay * dt# / to_lat


def RandomWalkDiffusion(particle, fieldset, time, dt):
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    dK = RK4(fieldset.dK_dx, fieldset.dK_dy, particle.lon, particle.lat, time, dt)
    half_dx = 0.5 * dK[0] * to_lon * dt
    half_dy = 0.5 * dK[1] * to_lat * dt
    Rand = np.random.uniform(0, 1.)
    K_at_half = RK4(fieldset.K, fieldset.K, particle.lon + half_dx, particle.lat + half_dy, time, dt)[0]
    R = Rand * K_at_half * dt #np.sqrt(4 * K_at_half * dt)
    angle = np.random.uniform(0, 2*np.pi)
    CorrectionX = dK[0] * dt * to_lon
    CorrectionY = dK[1] * dt * to_lat
    particle.Cx = CorrectionX
    particle.Cy = CorrectionY
    particle.Dx = R*np.cos(angle) * to_lon
    particle.Dy = R*np.sin(angle) * to_lat


def UndoMove(particle, fieldset, time, dt):
    print("UndoMove triggered! Moving particle")
    print("from: %s | %s" % (particle.lon, particle.lat))
    temp_lon = particle.lon
    temp_lat = particle.lat
    particle.lon -= particle.Ax + (particle.Dx + particle.Cx + particle.Vx)# * to_lon
    particle.lat -= particle.Ay + (particle.Dy + particle.Cy + particle.Vy)# * to_lat
    particle.Ax = particle.Ay = particle.Dx = particle.Dy = particle.Cx = particle.Cy = particle.Vx = particle.Vy = 0.0
    #particle.lon = 200
    #particle.lat = 0
    print("to:   %s | %s" % (particle.lon, particle.lat))
    if particle.lon == temp_lon and particle.lat == temp_lat:
        print("Positions are the same! Using particle saved previous positions...")
        particle.lon = particle.prev_lon
        particle.lat = particle.prev_lat


def MoveOffLand(particle, fieldset, time, dt):
    onland = fieldset.LandMask[0, particle.lon, particle.lat, particle.depth]
    if onland == 1:
        oldlon = particle.lon - particle.Ax - particle.Dx - particle.Cx - particle.Vx
        oldlat = particle.lat - particle.Ay - particle.Dy - particle.Cy - particle.Vy
        lat_convert = 1 / 1000. / 1.852 / 60.
        lon_convert = to_lat / math.cos(oldlat*math.pi/180)
        Kfield_new = fieldset.K[time, oldlon, oldlat, particle.depth]
        r_var_new = 1/3.
        Dx_component = math.sqrt(2 * Kfield_new * dt / r_var_new) * lon_convert
        Dy_component = math.sqrt(2 * Kfield_new * dt / r_var_new) * lat_convert
        count = 0
        particle.In_Loop = 0
        while onland > 0:
            #return ErrorCode.ErrorOutOfBounds
            #print("particle on land at %s|%s" % (particle.lon, particle.lat))
            particle.lon -= particle.Dx
            particle.lat -= particle.Dy
            Rx_new = random.uniform(-1., 1.)
            Ry_new = random.uniform(-1., 1.)
            particle.Dx = Dx_component * Rx_new
            particle.Dy = Dy_component * Ry_new
            particle.lon += particle.Dx
            particle.lat += particle.Dy
            onland = fieldset.LandMask[0, particle.lon, particle.lat, particle.depth]
            #print("attempting move to %s|%s" % (particle.lon, particle.lat))
            #print("onland now = %s" % onland)
            count += 1
            particle.In_Loop += 1
            if count > 500:
                particle.lon -= particle.Ax + (particle.Dx + particle.Cx + particle.Vx)# * to_lon
                particle.lat -= particle.Ay + (particle.Dy + particle.Cy + particle.Vy)# * to_lat
                particle.Ax = particle.Ay = particle.Dx = particle.Dy = particle.Cx = particle.Cy = particle.Vx = particle.Vy = 0.0
                onland = 0


# Kernel to call a generic particle update function
def Update(particle, fieldset, time, dt):
    particle.update()


def Move(particle, fieldset, time, dt):
    if particle.active == 1:
        #to_lat = 1 / 1000. / 1.852 / 60.
        #to_lon = to_lat / math.cos(particle.lat*math.pi/180)
        #print("Ax=%s Dx=%s Cx=%s Vx=%s dHdx=%s at time %s" % (particle.Ax , particle.Dx, particle.Cx, particle.Vx, particle.dHdx, time))
        #print("Ay=%s Dy=%s Cy=%s Vy=%s dHdy=%s at time %s" % (particle.Ay , particle.Dy, particle.Cy, particle.Vy, particle.dHdy, time))
        particle.prev_lon = particle.lon
        particle.prev_lat = particle.lat
        particle.lon += particle.Ax + (particle.Dx + particle.Cx + particle.Vx)# * to_lon
        particle.lat += particle.Ay + (particle.Dy + particle.Cy + particle.Vy)# * to_lat
        #particle.lon += particle.Ax + particle.Vx
        #particle.lat += particle.Ay + particle.Vy
        #particle.lon += Dx + Cx
        #particle.lat += Dy + Cy


# Some simple movement kernels for testing purposes:
def MoveEast(particle, fieldset, time, dt):
    if particle.active == 1:
        to_lat = 1 / 1000. / 1.852 / 60.
        to_lon = to_lat / math.cos(particle.lat*math.pi/180)
        particle.lon += 3 * dt * to_lon


def MoveWest(particle, fieldset, time, dt):
    if particle.active == 1:
        to_lat = 1 / 1000. / 1.852 / 60.
        to_lon = to_lat / math.cos(particle.lat*math.pi/180)
        particle.lon -= 3 * dt * to_lon


def RegionBound(particle, fieldset, time, dt):
    if particle.active == 1:
        if fieldset.minlon > particle.lon or fieldset.maxlon < particle.lon:
            particle.active = 0
        if fieldset.minlat > particle.lat or fieldset.maxlat < particle.lat:
            particle.active = 0