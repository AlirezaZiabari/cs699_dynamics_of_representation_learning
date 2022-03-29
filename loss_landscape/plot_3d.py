"""Plot the contours and trajectory give the corresponding files"""

import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot
import matplotlib.colors as colors
from matplotlib import animation

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')


class LogNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)


def animate(i):
    # this function is an internal auxiliary function for plotting gifs
    ax.view_init(elev=20., azim=2 * i)
    return fig,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument("--trajectory_file", required=False, default=None)
    parser.add_argument("--surface_file", required=False, default=None)
    parser.add_argument("--plot_prefix", required=True, help="prefix for the figure names")

    parser.add_argument("--x_uplim", required=True, help="x uplim for plotting", type=float, default=30)
    parser.add_argument("--x_lowlim", required=True, help="x lowlim for plotting", type=float, default=-10)
    parser.add_argument("--y_uplim", required=True, help="y uplim for plotting", type=float, default=30)
    parser.add_argument("--y_lowlim", required=True, help="y lowlim for plotting", type=float, default=-10)
    parser.add_argument("--zlim", required=True, help="zlim for plotting", type=int, default=15)
    parser.add_argument("--use_log_scale", action="store_true", default=False)

    args = parser.parse_args()

    loss_level_diff = 0.005
    log_alpha = -5
    N = 30

    if args.use_log_scale:
        result_folder = args.result_folder + '_log'
    else:
        result_folder = args.result_folder
        
    # set up logging
    os.makedirs(f"{result_folder}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.surface_file:
        # create a contour plot
        data = np.load(f"{args.surface_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        losses = data["losses"]
        if args.use_log_scale:
            losses = np.log(losses)
            
        acc = data["accuracies"]

        xcoords_within = xcoords[(args.x_uplim > xcoords) & (args.x_lowlim < xcoords)]
        ycoords_within = ycoords[(args.y_uplim > ycoords) & (args.y_lowlim < ycoords)]
        losses_within = losses[((args.x_uplim > xcoords) & (args.x_lowlim < xcoords)), :][:,
                        (args.y_uplim > ycoords) & (args.y_lowlim < ycoords)]

        X, Y = np.meshgrid(xcoords_within, ycoords_within, indexing="ij")
        Z = losses_within
        Z[Z > args.zlim] = args.zlim

        data_max, data_min = np.amax(Z), np.amin(Z)
        if args.use_log_scale:
            norm = colors.Normalize(vmin=-5, vmax=7)
        else:
            norm = colors.Normalize(vmin=0, vmax=args.zlim)
            # log_gamma = (np.log(data_max - data_min) - log_alpha) / N
            # levels = data_min + np.exp(log_alpha + log_gamma * np.arange(N + 1))
            # levels[0] = data_min
            # levels[-1] = data_max
            # norm = LogNormalize(1e-8, 7, log_alpha=log_alpha)

        zlim_up = np.min([args.zlim, np.max(Z)])
        
        if args.use_log_scale:
            zlim_low = np.min(Z)
        else:
            zlim_low = 0
        zlim_low_ticks = np.min(Z)
        
        origin_clip_tick = Z[np.argmin(np.abs(xcoords_within))][np.argmin(np.abs(ycoords_within))]
        origin_unclipped_tick = losses[np.argmin(np.abs(xcoords))][np.argmin(np.abs(ycoords))]
        CS = ax.plot_surface(X, Y, Z, cmap='jet_r', norm=norm, alpha=0.7)  

        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.set_zticks([zlim_low_ticks, zlim_up])
        ax.set_title(f"Loss at the origin: {origin_unclipped_tick:.2f}")

        ax.set_zlim(zlim_low, zlim_up)
        ax.view_init(elev=20., azim=0)
        fig.savefig(
            f"{result_folder}/{args.plot_prefix}_surface_2d_contour_level{loss_level_diff:.0e}_zlim{args.zlim}.jpg",
            dpi=300,
            bbox_inches='tight'
        )
        file_name = f"{result_folder}/{args.plot_prefix}_3d_contour_level{loss_level_diff:.0e}_zlim{args.zlim}.gif"

        anim = animation.FuncAnimation(fig, animate, frames=180, blit=True)
        anim.save(file_name, fps=360, dpi=200, writer='imagemagick')
