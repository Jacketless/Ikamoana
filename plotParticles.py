#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if (not(isinstance(line, list)) or not(isinstance(line[0],
                                           mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    color = line[0].get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line[0].get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows


def particleplotting(filename, psize, recordedvar, rcmap, backgroundfield, dimensions, cmap, drawland, limits, display,
                     start=1, mode='movie2d', output='particle_plot'):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    time = pfile.variables['time']
    #z = pfile.variables['z']
    #active = pfile.variables['active']

    if display is not 'none':
        title = pfile.variables[display]

    print(limits)

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
                day = int((i-1140)/8)
                plt.suptitle("Day %s" % day)#"Cohort age = %s months" % title[0,i])
            return scat,

        if drawland:
            m = Basemap(width=12000000, height=9000000, projection='cyl',
                        resolution='f', llcrnrlon=np.round(np.amin(lon)), llcrnrlat=np.amin(lat),
                        urcrnrlon=np.amax(lon), urcrnrlat=np.amax(lat), area_thresh = 10)

        anim = animation.FuncAnimation(fig, animate, frames=np.arange(start, lon.shape[1]),
                                       interval=1, blit=False)
        plt.show()

    elif mode == 'to_file':
        fig = plt.figure(1)
        ax = plt.axes(xlim=[limits[0], limits[1]], ylim=[limits[2], limits[3]])
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        indices = np.rint(np.linspace(0, len(lon)-1, 100)).astype(int)
        scat = ax.scatter(lon[indices, 0], lat[indices, 0], s=psize, c='black')
        if drawland:
            m = Basemap(width=12000000, height=9000000, projection='cyl',
                        resolution='f', llcrnrlon=np.round(np.amin(lon)), llcrnrlat=np.amin(lat),
                        urcrnrlon=np.amax(lon), urcrnrlat=np.amax(lat), area_thresh = 10)

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
                day = int((i-1140)/8)
                #plt.suptitle("Day %s" % day)
                plt.suptitle("Cohort age = %s months" % title[0,i])
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

    args = p.parse_args()

    if args.background is not 'none':
        args.background = {args.variable: args.background}

    if args.size is 'none':
        if args.mode is 'movie2d':
            psize = 60
        else:
            psize = 1
    else:
        psize = int(args.size)

    particleplotting(args.particlefile, psize, args.recordedvar, args.colourmap_recorded, args.background, args.dimensions, args.colourmap, args.landdraw,
                     args.limits, args.title,start=args.start_time, mode=args.mode, output=args.output)
