import argparse
import pathlib
import sys
import numpy as np
import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt  # for plotting
import pandas as pd

from chrest.chrestData import ChrestData
from xdmfGenerator import XdmfGenerator
from supportPaths import expand_path
import ablateData

plt.rcParams["font.family"] = "Noto Serif CJK JP"


class VTcpData:

    def __init__(self, files=None, fields=None):
        self.prf = None
        self.field_size = len(fields)

        # Initialize the
        vtcp = ablateData.AblateData(files)
        [data_tmp, times_tmp, names_tmp] = vtcp.get_field(fields[0])
        self.data = np.zeros((self.field_size, np.shape(data_tmp)[0], np.shape(data_tmp)[1]))
        self.rgb = np.zeros((np.shape(data_tmp)[0], np.shape(data_tmp)[1], self.field_size))
        self.times = np.zeros(self.field_size)
        self.names = np.zeros(self.field_size)
        self.tcp_temperature = None
        coords_tmp = vtcp.compute_cell_centers(3)
        self.coords = np.zeros((self.field_size, np.shape(coords_tmp)[0], np.shape(coords_tmp)[1]))

        for f in range(self.field_size):
            [self.data[f, :, :], self.times[f], self.names[f]] = vtcp.get_field(fields[f])
            self.coords[f, :, :] = vtcp.compute_cell_centers(3)

        self.set_limits()  # Sets the time step range of the processing

    def get_tcp_temperature(self):
        for n in range(len(self.data[0, :, 0])):
            # Calculate the two-color-pyrometry temperature of the frame
            # First, get the intensity ratio between the red and green channels (0 and 1)
            # Then, use the ratio to get the temperature
            # Finally, plot the temperature
            ratio = self.data[1, n, :] / self.data[0, n, :]

            c = 3.e8  # Speed of light
            h = 6.626e-34  # Planck's constant
            k = 1.3806e-23  # Boltzmann Constant

        # Planck's first and second constant
        c1 = 2. * np.pi * h * c * c
        c2 = h * c / k

        lambdaR = 650e-9
        lambdaG = 532e-9
        tcp_temperature = np.zeros([len(ratio)], dtype=np.dtype(float))
        
        threshold_fraction = 0.05  # Threshold for the absolute intensity (keep at 0.15?)
        
        for i in range(len(ratio)):
            if self.data[0, n, i] < threshold_fraction * np.max(self.data[0, n, :]) or self.data[1, n, i] < threshold_fraction * np.max(self.data[1, n, :]):
                tcp_temperature[i] = 0  # If either channel is zero, set the temperature to zero
            else:
                tcp_temperature[i] = (c2 * ((1. / lambdaR) - (1. / lambdaG))) / (
                        np.log(ratio[i]) + np.log((lambdaG / lambdaR) ** 5))  # + np.log(4.24 / 4.55))
            if tcp_temperature[i] < 300:  # or tcp_temperature[i] > 3500:
                tcp_temperature[i] = 300
        # 4.24 and 4.55 are empirical optical constants for refractive index for the red and green channels
                self.tcp_temperature[n].append(tcp_temperature)
        # return self.tcp_temperature

        # Get the size of a single mesh.
        # Iterate through the time steps
        # Iterate through each time step and place a point on the plot
    def plot_temperature_step(self, n, name):
        # self.get_tcp_temperature()  # Calculate the TCP temperature of the given boundary intenisties
        # TODO: Fix the indexing for getting the TCP temperature
        tcp_temperature_frame = np.vstack((self.coords[0, :, 0], self.coords[0, :, 1], self.tcp_temperature[:]))
        tcp_temperature_frame = np.transpose(tcp_temperature_frame)

        d = pd.DataFrame(tcp_temperature_frame, columns=['x', 'y', 'd'])
        D = d.pivot_table(index='x', columns='y', values='d').T.values

        X_unique = np.sort(d.x.unique())
        Y_unique = np.sort(d.y.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        CS = ax.imshow(D, interpolation='none', cmap="inferno",
                       origin='lower',
                       extent=[tcp_temperature_frame[:, 0].min(), tcp_temperature_frame[:, 0].max(),
                               tcp_temperature_frame[:, 1].min(), tcp_temperature_frame[:, 1].max()],
                       vmax=4500, vmin=300)
        fig.colorbar(CS, shrink=0.5, pad=0.05)
        # ax.clabel(CS, inline=True, fontsize=10)
        # ax.set_title('CHREST Format vTCP (n = ' + str(n) + ')')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # ax.legend(r"Temperature $[K]$")  # Add label for the temperature
        plt.savefig(str(name) + "." + str(n).zfill(3) + ".png", dpi=1000, bbox_inches='tight')
        plt.show()

        # tcp_temperature_filtered = tcp_temperature[tcp_temperature < 3500]
        # tcp_temperature_filtered = tcp_temperature_filtered[300 < tcp_temperature_filtered]
        # print(np.mean(tcp_temperature_filtered))


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
        exposure_fraction = 1.0
        brightness_max = np.array([-100.0, -100.0, -100.0])
        for fieldIndex in range(self.field_size):
            for timeStep in range(np.shape(self.data)[1]):
                for pointIndex in range(np.shape(self.data)[2]):
                    brightness_transformed = np.log(np.pi * self.data[fieldIndex, timeStep, pointIndex] * deltaT)
                    if brightness_transformed > brightness_max[fieldIndex]:
                        brightness_max[fieldIndex] = brightness_transformed
        for fieldIndex in range(self.field_size):
            prf_row_max = int(255.0 * exposure_fraction)
            shift_constant = self.prf[prf_row_max, fieldIndex] - brightness_max[fieldIndex]

        for fieldIndex in range(self.field_size):
            for timeStep in range(np.shape(self.data)[1]):
                for pointIndex in range(np.shape(self.data)[2]):
                    brightness = 0
                    brightness_transformed = np.log(np.pi * self.data[fieldIndex, timeStep, pointIndex] * deltaT)
                    brightness_transformed += shift_constant

                    if np.isinf(brightness_transformed):
                        brightness_transformed = 0
                    for brightnessIndex in range(np.shape(self.prf)[0]):
                        if self.prf[brightnessIndex, fieldIndex] > brightness_transformed:
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


def get_uncertainty_field(vtcp_data, chrest_data):
    # Get the TCP temperature from the boundary information
    tcp_temperature = vtcp_data.get_tcp_temperature(i)

    # Now that we have the tcp temperature, we want to get the maximum temperatures in each of the ray lines.
    dns_temperature, _, _ = chrest_data.get_field("temperature")

    # Iterate along the ray direction for each cell in the plane of the boundary
    # Need boundary information to do this

    # Get the difference between the DNS temperature and the TCP temperature
    uncertainty_field = dns_temperature - vtcp_data.tcp_temperature

    return uncertainty_field


def plot_uncertainty_field(uncertainty_field):
    return None


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
    parser.add_argument('--name', dest='name', type=str,
                        help='Which index to finish the data processing.', required=True)

    args = parser.parse_args()

    vTCP = VTcpData(args.hdf5_file, args.fields)  # Initialize the virtual TCP creation.

    # vTCP.rgb_transform(args.deltaT)
    # vTCP.plot_rgb_step(41, "vTCP_RGB_ignition")
    print(len(vTCP.data[0, :, 0]))

    for i in range(len(vTCP.data[0, :, 0])):
        #     vTCP.plot_rgb_step(i, "vTCP_RGB_ignition")
        vTCP.plot_temperature_step(i, args.name)

    # Get the CHREST data associated with the simulation for the 3D stuff
    file = []  # The files should be input here from the input arguments
    data_3d = ChrestData(file)

    # Calculate the difference between the DNS temperature and the tcp temperature
    temp_uncert_field = get_uncertainty_field(vTCP, data_3d)
    plot_uncertainty_field(temp_uncert_field)

    # It would be worth correlating the uncertainty field to something non-dimensional
    # Or at least related to the flame structure so that it can be generalized
    # Maybe the mixture fraction is an appropriate value?    # Save mp4 out of all the frames stitched together.

    print('Done')
