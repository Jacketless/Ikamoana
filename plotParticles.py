#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def getMFRegion(lon, lat):
    region = -1
    if lon > 210: # Longitudinally outside WCPO assessment area
        region = 6
    else:
        if lat > 50: # Latitudinally outside assessment area
            region = 6
        elif lat < -20:
            region = 6
        else:
            if lat > 20:
                region = 1
            else: # regions 2 to 5
                if lon < 140:
                    region = 4
                elif lon > 170:
                    region = 3
                else: # regions 2 and 5 (the tricky one)
                    if lat > 0:
                        region = 2
                    elif lon > 160:
                        region = 2
                    else:
                        if lon > 155 and lat > -5:
                            region = 2
                        else:
                            region = 5

    return region


def particleplotting(filename, psize, recordedvar, rcmap, backgroundfield, dimensions, cmap, drawland, limits, display,
                     start=1, mode='movie2d', plot_mfregion=False, mfregion_start=1, mf_focus=0, output='particle_plot'):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    time = pfile.variables['time']
    #z = pfile.variables['z']
    #active = pfile.variables['active']

    if display is not 'none':
        title = pfile.variables[display]

    if plot_mfregion:
        mf_cols = []
        colmap = {1: 'red', 2: 'blue', 3: 'orange', 4: 'green', 5: 'purple', 6: 'grey'}
        if mf_focus == 0:
            for p in range(lon.shape[0]):
                mf_cols.append(colmap[getMFRegion(lon[p,mfregion_start], lat[p,mfregion_start])])
        else:
            for p in range(lon.shape[0]):
                r = getMFRegion(lon[p,mfregion_start], lat[p,mfregion_start])
                mf_cols.append(colmap[r] if r is mf_focus else 'lightgrey')

    if limits is -1:
        limits = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)]

    if recordedvar is not 'none':
        print('Particles coloured by: %s' % recordedvar)
        rMin = np.min(pfile.variables[recordedvar])
        rMax = np.max(pfile.variables[recordedvar])
        print('Min = %f, Max = %f' % (rMin, rMax))
        record = (pfile.variables[recordedvar]-rMin)/(rMax-rMin)

    if backgroundfield is not 'none':
        bfile = Dataset(backgroundfield.values()[0], 'r')
        bX = bfile.variables[dimensions[0]]
        bY = bfile.variables[dimensions[1]]
        bT = bfile.variables[dimensions[2]]
        # Find the variable that exists across at least two spatial and one time dimension
        if backgroundfield.keys()[0] is 'none':
            def checkShape(var):
                print(np.shape(var))
                if len(np.shape(var)) > 2:
                    return True

            for v in bfile.variables:
                if checkShape(bfile.variables[v]):
                    bVar = bfile.variables[v]
                    print('Background variable is %s' % v)
                    break
        else:
            bVar = bfile.variables[backgroundfield.keys()[0]]

    if mode == '3d':
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        for p in range(len(lon)):
            ax.plot(lon[p, :], lat[p, :], z[p, :], '.-')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth')
        plt.show()
    elif mode == '2d':
        fig, ax = plt.subplots(1, 1)
        #subsample
        indices = np.rint(np.linspace(0, len(lon)-1, 100)).astype(int)
        #indices = range(len(lon[:,0]))
        if backgroundfield is not 'none':
            time=18
            plt.contourf(bX[:], bY[:], bVar[time, 0, :, :], zorder=-1, vmin=0, vmax=np.max(bVar[time, 0, :, :]),
                         levels=np.linspace(0, np.max(bVar[time, 0, :, :]), 100), xlim=[limits[0], limits[1]],
                         ylim=[limits[2], limits[3]], cmap=cmap)
            #plt.plot(np.transpose(lon[indices,:]), np.transpose(lat[indices,:]), '.-', linewidth=psize,
            #         markersize=psize, c='blue')
            plt.quiver(np.transpose(lon[indices,:-1]), np.transpose(lat[indices,:-1]),
                       np.transpose(lon[indices,1:])-np.transpose(lon[indices,:-1]),
                       np.transpose(lat[indices,1:])-np.transpose(indices,lat[:-1]),
                       scale_units='xy', angles='xy', scale=1)
            plt.xlim([limits[0], limits[1]])
            plt.ylim([limits[2], limits[3]])
        else:
            lines = ax.plot(np.transpose(lon[indices,:]), np.transpose(lat[indices,:]), 'k-', linewidth=psize,
                     markersize=psize, c='white')
            plt.xlim([limits[0], limits[1]])
            plt.ylim([limits[2], limits[3]])
            add_arrow_to_line2D(ax, lines, arrow_locs=np.linspace(0., 1., 200), arrowstyle='->')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        if drawland:
            m = Basemap(width=12000000, height=9000000, projection='cyl',
                        resolution='f', llcrnrlon=limits[0], llcrnrlat=limits[2],
                        urcrnrlon=limits[1], urcrnrlat=limits[3], epsg=4714, area_thresh = 0.1)
            m.drawcoastlines()
            m.etopo()
            #m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
            m.fillcontinents(color='forestgreen', lake_color='aqua')
        plt.show()

    elif mode == 'movie2d':

        fig = plt.figure(1)
        ax = plt.axes(xlim=[limits[0], limits[1]], ylim=[limits[2], limits[3]])
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        indices = np.rint(np.linspace(0, len(lon)-1, 100)).astype(int)
        scat = ax.scatter(lon[indices, 0], lat[indices, 0], s=psize, c='black')
        # Offline calc contours still to do
        def animate(i):
            ax.cla()
            #active_list = active[:, i] > 0
            active_list = range(len(lon[:,i]))#indices

            if drawland:
                m.drawcoastlines()
                m.fillcontinents(color='forestgreen', lake_color='aqua')
            if recordedvar is not 'none':
                scat = ax.scatter(lon[active_list, i], lat[active_list, i], s=psize, c=record[active_list, i],
                                  cmap=rcmap, vmin=0, vmax=1)
            elif plot_mfregion:
                scat = ax.scatter(lon[active_list, i], lat[active_list, i], s=psize, color=mf_cols,
                                  cmap='summer', vmin=0, vmax=1)
            else:
                scat = ax.scatter(lon[active_list, i], lat[active_list, i], s=psize, c='blue', edgecolors='black')
            ax.set_xlim([limits[0], limits[1]])
            ax.set_ylim([limits[2], limits[3]])
            if backgroundfield is not 'none':
                field_time = np.argmax(bT > time[0, i]) - 1
                plt.contourf(bY[:], bX[:], bVar[field_time, :, :], vmin=0, vmax=np.max(bVar[field_time, :, :]),
                             levels=np.linspace(0, np.max(bVar[field_time, :, :]), 100), xlim=[limits[0], limits[1]],
                             zorder=-1,ylim=[limits[2], limits[3]], cmap=cmap)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            if display is not 'none':
                plt.suptitle("Cohort age %s months" % display)#"Cohort age = %s months" % title[0,i])
                plt.title("Region %s" % mf_focus)
            return scat,

        if drawland:
            m = Basemap(width=12000000, height=9000000, projection='cyl',
                        resolution='c', llcrnrlon=np.round(np.amin(lon)), llcrnrlat=np.amin(lat),
                        urcrnrlon=np.amax(lon), urcrnrlat=np.amax(lat), area_thresh = 10)

        anim = animation.FuncAnimation(fig, animate, frames=np.arange(start, lon.shape[1]),
                                       interval=1, blit=False)
        plt.show()

    elif mode == 'to_file':
        fig = plt.figure(1, figsize=(16, 8), dpi=100)
        ax = plt.axes(xlim=[limits[0], limits[1]], ylim=[limits[2], limits[3]])
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        indices = np.rint(np.linspace(0, len(lon)-1, 100)).astype(int)
        scat = ax.scatter(lon[indices, 0], lat[indices, 0], s=psize, c='black')
        if drawland:
            m = Basemap(width=12000000, height=9000000, projection='cyl',
                        resolution='c', llcrnrlon=np.round(np.amin(lon)), llcrnrlat=np.amin(lat),
                        urcrnrlon=np.amax(lon),  urcrnrlat=np.amax(lat), area_thresh = 10)

        for i in np.arange(start, lon.shape[1]):
            ax.cla()
            #active_list = active[:, i] > 0
            active_list = range(len(lon[:,i]))#indices

            if drawland:
                m.drawcoastlines()
                m.fillcontinents(color='forestgreen', lake_color='aqua')
            if recordedvar is not 'none':
                scat = ax.scatter(lon[active_list, i], lat[active_list, i], s=psize, c=record[active_list, i],
                                  cmap=rcmap, vmin=0, vmax=1)
            elif plot_mfregion:
                scat = ax.scatter(lon[active_list, i], lat[active_list, i], s=psize, color=mf_cols,
                                  cmap='summer', vmin=0, vmax=1)
            else:
                scat = ax.scatter(lon[active_list, i], lat[active_list, i], s=psize, c='blue', edgecolors='black')
            ax.set_xlim([limits[0], limits[1]])
            ax.set_ylim([limits[2], limits[3]])
            if backgroundfield is not 'none':
                field_time = np.argmax(bT > time[0, i]) - 1
                plt.contourf(bY[:], bX[:], bVar[field_time, :, :], vmin=0, vmax=np.max(bVar[field_time, :, :]),
                             levels=np.linspace(0, np.max(bVar[field_time, :, :]), 100), xlim=[limits[0], limits[1]],
                             zorder=-1,ylim=[limits[2], limits[3]], cmap=cmap)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            if display is not 'none':
                plt.suptitle("Region %s\nCohort age %s months" % (mf_focus if mf_focus != 0 else 'All', title[0,i]))
            plt.savefig('Plots/%s%s.png' % (output, i))


if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d', 'movie2d', 'to_file'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-r', '--recordedvar', type=str, default='none',
                   help='Name of a variable recorded along trajectory')
    p.add_argument('-b', '--background', type=str, default='none',
                   help='Name of file containing background field to display')
    p.add_argument('-v', '--variable', type=str, default='none',
                   help='Name of variable to display in background field')
    p.add_argument('-c', '--colourmap', type=str, default='jet',
                   help='Colourmap for field data')
    p.add_argument('-cr', '--colourmap_recorded', type=str, default='autumn',
                   help='Colourmap for particle recorded data')
    p.add_argument('-s', '--size', type=str, default='none',
                   help='Size of drawn particles and tracks')
    p.add_argument('-ld', '--landdraw', type=bool, default=False,
                   help='Boolean for whether to draw land using mpl.basemap package')
    p.add_argument('-l', '--limits', type=float, nargs=4, default=-1,
                   help='Limits for plotting, given min_lon, max_lon, min_lat, max_lat')
    p.add_argument('-d', '--dimensions', type=str, nargs=3, default=['nav_lon', 'nav_lat', 'time_counter'],
                   help='Name of background field dimensions in order of lon, lat, and time')
    p.add_argument('-t', '--title', type=str, default='none',
                   help='Variable to pull from particle file and display during movie')
    p.add_argument('-st', '--start_time', type=int, default=1,
                   help='Timestep from which to begin animation')
    p.add_argument('-o', '--output', type=str, default='particle_plot',
                   help='Filename stem when writing to file')
    p.add_argument('-mf', '--mf_colours', type=str, default='False',
                   help='Boolean flag for colouring particles by SKJ assessment region')
    p.add_argument('-mff', '--mf_focus', type=int, default=0,
                   help='which single region to be coloured, defaults to zero which is all regions')
    p.add_argument('-mfs', '--mf_start', type=int, default=0,
                   help='Timestep at which to define mf region classification, defaults to the first')

    args = p.parse_args()

    if args.background is not 'none':
        args.background = {args.variable: args.background}
    if args.mf_colours == 'True':
        print("MF")
        mf_colours = True
    else:
        print("NO MF")
        mf_colours = False

    if args.size is 'none':
        if args.mode is 'movie2d':
            psize = 60
        else:
            psize = 1
    else:
        psize = int(args.size)

    particleplotting(args.particlefile, psize, args.recordedvar, args.colourmap_recorded, args.background, args.dimensions, args.colourmap, args.landdraw,
                     args.limits, args.title,start=args.start_time, mode=args.mode, output=args.output,
                     plot_mfregion=mf_colours, mfregion_start=args.mf_start, mf_focus=args.mf_focus)
