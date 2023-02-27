import argparse
import pathlib
import sys
import numpy as np
import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt  # for plotting
import pandas as pd
from xdmfGenerator import XdmfGenerator
from supportPaths import expand_path
import ablateData


class vTCPData:

    def __init__(self, files=None, fields=None):
        self.fieldSize = len(fields)

        # Initialize the
        vtcp = ablateData.AblateData(files)
        [data_tmp, times_tmp, names_tmp] = vtcp.get_field(fields[0])
        self.data = np.zeros((self.fieldSize, np.shape(data_tmp)[0], np.shape(data_tmp)[1]))
        self.rgb = np.zeros((np.shape(data_tmp)[0], np.shape(data_tmp)[1], self.fieldSize))
        self.times = np.zeros(self.fieldSize)
        self.names = np.zeros(self.fieldSize)
        coords_tmp = vtcp.compute_cell_centers(3)
        self.coords = np.zeros((self.fieldSize, np.shape(coords_tmp)[0], np.shape(coords_tmp)[1]))

        for f in range(self.fieldSize):
            [self.data[f, :, :], self.times[f], self.names[f]] = vtcp.get_field(fields[f])
            self.coords[f, :, :] = vtcp.compute_cell_centers(3)

        self.set_limits()  # Sets the time step range of the processing

    # Get the size of a single mesh.
    # Iterate through the time steps
    # Iterate through each time step and place a point on the plot
    def plot_intensity_step(self, n, f):
        frame = np.vstack((self.coords[f, :, 0], self.coords[f, :, 1], self.data[f, n, :]))
        frame = np.transpose(frame)

        d = pd.DataFrame(frame, columns=['x', 'y', 'd'])
        D = d.pivot_table(index='x', columns='y', values='d').T.values

        X_unique = np.sort(d.x.unique())
        Y_unique = np.sort(d.y.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        CS = ax.imshow(D, interpolation='bilinear', cmap="inferno",
                       origin='lower',
                       extent=[frame[:, 0].min(), frame[:, 0].max(), frame[:, 1].min(), frame[:, 1].max()],
                       vmax=abs(D).max(), vmin=-abs(D).max())
        # ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('CHREST Format vTCP (n = ' + str(n) + ')')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig('vTCP_test', dpi=1000, bbox_inches='tight')
        plt.show()

    def set_limits(self):
        if args.n_end:
            self.end = args.n_end
        else:
            self.end = len(self.data[:, 0]) - 1  # Set the end time to the last step by default
        if args.n_start:
            self.start = args.n_start
        else:
            self.start = 0  # Set the start time step to the first by default

    def rgb_transform(self, deltaT):
        self.prf = np.loadtxt("PRF_Color.csv", delimiter=',', skiprows=0)

        for fieldIndex in range(self.fieldSize):
            for timeStep in range(np.shape(self.data)[1]):
                for pointIndex in range(np.shape(self.data)[2]):
                    for brightnessIndex in range(len(self.prf)):
                        brightnessTransformed = 1  # np.log(self.data[fieldIndex, timeStep, pointIndex] * deltaT)
                        if (self.prf[brightnessIndex, fieldIndex] > brightnessTransformed):
                            if (fieldIndex == 0):
                                brightness = brightnessIndex / 255
                            break
                    self.rgb[
                        timeStep, pointIndex, fieldIndex] = brightness  # assign the pixel brightness based on camera prf

    def plot_rgb_step(self, n):
        frame = np.vstack(
            (self.coords[0, :, 0], self.coords[0, :, 1], self.rgb[n, :, 0], self.rgb[n, :, 1], self.rgb[n, :, 2]))
        frame = np.transpose(frame)

        d = pd.DataFrame(frame, columns=['x', 'y', 'r', 'g', 'b'])
        D = d.pivot_table(index='x', columns='y', values=['r', 'g', 'b']).T.values

        X_unique = np.sort(d.x.unique())
        Y_unique = np.sort(d.y.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        CS = ax.imshow(self.rgb, interpolation='bilinear',  # cmap="inferno",
                       origin='lower',
                       extent=[frame[:, 0].min(), frame[:, 0].max(), frame[:, 1].min(), frame[:, 1].max()],
                       vmax=abs(D).max(), vmin=-abs(D).max())
        # ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('CHREST Format vTCP (n = ' + str(n) + ')')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig('vTCP_test', dpi=1000, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate plots from virtual TCP data. '
                    'See https://github.com/cet-lab/experimental-post-processing/wiki'
                    '/Matlab-To-XdmfGenerator for details.  ')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The path to the ablate hdf5 file(s) containing the ablate data.'
                             '  A wild card can be used '
                             'to supply more than one file.')
    parser.add_argument('--fields', dest='fields', type=str, nargs="+",
                        help='List of intensity fields in RGB order.', required=True)
    parser.add_argument('--start', dest='n_start', type=int,
                        help='Which index to start the data processing.')
    parser.add_argument('--end', dest='n_end', type=int,
                        help='Which index to finish the data processing.')
    parser.add_argument('--exposure_time', dest='deltaT', type=float,
                        help='Which index to finish the data processing.', required=True)

    args = parser.parse_args()

    vTCP = vTCPData(args.hdf5_file, args.fields)  # Initialize the virtual TCP creation.

    # vTCP.plot_step(5)
    vTCP.rgb_transform(args.deltaT)
    vTCP.plot_rgb_step(5)

    # Save mp4 out of all the frames stiched together.
    print('Done')
