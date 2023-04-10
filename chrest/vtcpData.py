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

plt.rcParams["font.family"] = "Noto Serif CJK JP"


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
    def plot_temperature_step(self, n, name):

        # Calculate the two-color-pyrometry temperature of the frame
        # First, get the intensity ratio between the red and green channels (0 and 1)
        # Then, use the ratio to get the temperature
        # Finally, plot the temperature
        ratio = self.data[0, n, :] / self.data[1, n, :]

        c = 3.e8  # Speed of light
        h = 6.626e-34  # Planck's constant
        k = 1.3806e-23  # Boltzmann Constant

        # Planck's first and second constant
        c1 = 2. * np.pi * h * c * c
        c2 = h * c / k

        lambdaR = 650e-9
        lambdaG = 532e-9
        tcp_temperature = np.zeros([len(ratio)], dtype=np.dtype(float))
        for i in range(len(ratio)):
            if self.data[0, n, i] == 0 or self.data[1, n, i] == 0:
                tcp_temperature[i] = 0  # If either channel is zero, set the temperature to zero
            else:
                tcp_temperature[i] = (c2 * ((1. / lambdaR) - (1. / lambdaG))) / (
                        np.log(ratio[i]) + np.log((lambdaG / lambdaR) ** 6) + np.log(4.24 / 4.55))
            if tcp_temperature[i] < 0:
                tcp_temperature[i] = 0
        # 4.24 and 4.55 are empirical optical constants for refractive index for the red and green channels
        # TODO: Why is float seen as NoneType??

        tcp_temperature_frame = np.vstack((self.coords[0, :, 0], self.coords[0, :, 1], tcp_temperature[:]))
        tcp_temperature_frame = np.transpose(tcp_temperature_frame)

        d = pd.DataFrame(tcp_temperature_frame, columns=['x', 'y', 'd'])
        D = d.pivot_table(index='x', columns='y', values='d').T.values

        X_unique = np.sort(d.x.unique())
        Y_unique = np.sort(d.y.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        CS = ax.imshow(D, interpolation='lanczos', cmap="inferno",
                       origin='lower',
                       extent=[tcp_temperature_frame[:, 0].min(), tcp_temperature_frame[:, 0].max(),
                               tcp_temperature_frame[:, 1].min(), tcp_temperature_frame[:, 1].max()],
                       vmax=abs(D).max(), vmin=-abs(D).max())
        # ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('CHREST Format vTCP (n = ' + str(n) + ')')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # ax.legend(r"Temperature $[K]$")  # Add label for the temperature
        # plt.savefig(name, dpi=1000, bbox_inches='tight')
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

        # Get the correct exposure for the camera by getting the maximum intensity for each channel and shifting to 255
        exposureFraction = 1.0
        brightnessMax = np.array([-100.0, -100.0, -100.0])
        for fieldIndex in range(self.fieldSize):
            for timeStep in range(np.shape(self.data)[1]):
                for pointIndex in range(np.shape(self.data)[2]):
                    brightnessTransformed = np.log(np.pi * self.data[fieldIndex, timeStep, pointIndex] * deltaT)
                    if brightnessTransformed > brightnessMax[fieldIndex]:
                        brightnessMax[fieldIndex] = brightnessTransformed
        for fieldIndex in range(self.fieldSize):
            prfRowMax = int(255.0 * exposureFraction)
            shiftConstant = self.prf[prfRowMax, fieldIndex] - brightnessMax[fieldIndex]

        for fieldIndex in range(self.fieldSize):
            for timeStep in range(np.shape(self.data)[1]):
                for pointIndex in range(np.shape(self.data)[2]):
                    brightness = 0
                    brightnessTransformed = np.log(np.pi * self.data[fieldIndex, timeStep, pointIndex] * deltaT)
                    brightnessTransformed += shiftConstant

                    if (np.isinf(brightnessTransformed)):
                        brightnessTransformed = 0
                    for brightnessIndex in range(np.shape(self.prf)[0]):
                        if self.prf[brightnessIndex, fieldIndex] > brightnessTransformed:
                            brightness = brightnessIndex / 255
                            break
                    self.rgb[timeStep, pointIndex, fieldIndex] = brightness  # pixel brightness based on camera prf

    def plot_rgb_step(self, n, name):
        rframe = np.vstack(
            (self.coords[0, :, 0], self.coords[0, :, 1], self.rgb[n, :, 0]))
        rframe = np.transpose(rframe)
        r = pd.DataFrame(rframe, columns=['x', 'y', 'r'])
        R = r.pivot_table(index='x', columns='y', values=['r']).T.values

        gframe = np.vstack(
            (self.coords[0, :, 0], self.coords[0, :, 1], self.rgb[n, :, 1]))
        gframe = np.transpose(gframe)
        g = pd.DataFrame(gframe, columns=['x', 'y', 'g'])
        G = g.pivot_table(index='x', columns='y', values=['g']).T.values

        bframe = np.vstack(
            (self.coords[0, :, 0], self.coords[0, :, 1], self.rgb[n, :, 2]))
        bframe = np.transpose(bframe)
        b = pd.DataFrame(bframe, columns=['x', 'y', 'b'])
        B = b.pivot_table(index='x', columns='y', values=['b']).T.values

        X_unique = np.sort(r.x.unique())
        Y_unique = np.sort(r.y.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        CS = ax.imshow(np.rot90(np.array([R.data, G.data, B.data]).T, axes=(0, 1)), interpolation='lanczos',
                       extent=[rframe[:, 0].min(), rframe[:, 0].max(), rframe[:, 1].min(), rframe[:, 1].max()],
                       vmax=abs(R).max(), vmin=-abs(R).max())
        # ax.clabel(CS, inline=True, fontsize=10)
        # ax.set_title('Simulated Camera (n = ' + str(n) + ')')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig(str(name) + "." + str(n).zfill(3) + ".png", dpi=1000, bbox_inches='tight')
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

    vTCP.rgb_transform(args.deltaT)
    # vTCP.plot_rgb_step(41, "vTCP_RGB_ignition")
    # for i in range(len(vTCP.data[0, :, 0])):
    #     vTCP.plot_rgb_step(i, "vTCP_RGB_ignition")
    vTCP.plot_temperature_step(41, "vTCP_temperature_ignition")
    #
    # Save mp4 out of all the frames stitched together.
    print('Done')
