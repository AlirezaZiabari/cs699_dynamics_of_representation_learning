"""Plot the contours and trajectory give the corresponding files"""

import argparse
import logging
import os

import numpy as np
from matplotlib import pyplot
import matplotlib.colors as colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument("--trajectory_file", required=False, default=None)
    parser.add_argument("--surface_file", required=False, default=None)
    parser.add_argument("--plot_prefix", required=True, help="prefix for the figure names")

    args = parser.parse_args()

    loss_level_diff = 0.005
    log_alpha = -5
    N = 30
    
    # set up logging
    os.makedirs(f"{args.result_folder}", exist_ok=True)
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
        acc = data["accuracies"]
        
        X, Y = np.meshgrid(xcoords, ycoords, indexing="ij")
        Z = losses
        data_max, data_min = np.amax(Z), np.amin(Z)
        fig = pyplot.figure()
        
        log_gamma = (np.log(data_max- data_min) - log_alpha) / N
        levels = data_min + np.exp(log_alpha + log_gamma * np.arange(N + 1))
        levels[0] = data_min
        levels[-1] = data_max
        
        norm = colors.LogNorm(0.02, 500, clip=False)
        
        CS = pyplot.contourf(X, Y, Z, cmap='jet_r', levels=levels, norm=norm) 
        colorbar = pyplot.colorbar(format='%.2f')
        pyplot.xlabel('PCA 1 Coefficient')
        pyplot.ylabel('PCA 2 Coefficient')
    
        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_surface_2d_contour_level{loss_level_diff:.0e}", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()

    if args.trajectory_file:
        # create a 2D plot of trajectory
        data = np.load(f"{args.trajectory_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]

        fig = pyplot.figure()
        pyplot.plot(xcoords, ycoords, linewidth=0.5, alpha=0.3)
        pyplot.scatter(xcoords, ycoords, marker='.', c=np.arange(len(xcoords)))
        pyplot.colorbar()   
        pyplot.tick_params('y', labelsize='x-large')
        pyplot.tick_params('x', labelsize='x-large')

        pyplot.xlabel('PCA 1 Projection Coefficient')
        pyplot.ylabel('PCA 2 Projection Coefficient')
    
        
        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_trajectory_2d", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()

    if args.surface_file and args.trajectory_file:
        # create a contour plot
        data = np.load(f"{args.surface_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        losses = data["losses"]
        acc = data["accuracies"]

        X, Y = np.meshgrid(xcoords, ycoords, indexing="ij")
        Z = losses
        
        data_max, data_min = np.amax(Z), np.amin(Z)
        fig = pyplot.figure()
        
        log_gamma = (np.log(data_max- data_min) - log_alpha) / N
        levels = data_min + np.exp(log_alpha + log_gamma * np.arange(N + 1))
        levels[0] = data_min
        levels[-1] = data_max

        norm = colors.LogNorm(0.02, 500, clip=False)
        CS = pyplot.contourf(X, Y, Z, cmap='jet_r', levels=levels, norm=norm) 
        colorbar = pyplot.colorbar(CS, format='%.2f')

        data = np.load(f"{args.trajectory_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        pyplot.plot(xcoords, ycoords, linewidth=0.5, alpha=0.3)
        TJ = pyplot.scatter(xcoords, ycoords, marker='.', c=np.arange(len(xcoords)))
        pyplot.tick_params('y', labelsize='x-large')
        pyplot.tick_params('x', labelsize='x-large')
    
        pyplot.xlabel('PCA 1 Coefficient')
        pyplot.ylabel('PCA 2 Coefficient')
    
        # colorbar = pyplot.colorbar(TJ)
        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_trajectory+contour_2d_level{loss_level_diff:.0e}", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()
