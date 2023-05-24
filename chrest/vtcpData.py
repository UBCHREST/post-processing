import argparse
import numpy as np
import matplotlib.pyplot as plt  # for plotting
import pandas as pd
import matplotlib.gridspec as gridspec
from chrest.chrestData import ChrestData
import os

plt.rcParams["font.family"] = "Noto Serif CJK JP"


class VTcpData:

    def __init__(self, files=None, fields=None, tcp_axis=None, base_path=None, write_path=None, save=None):
        h = 6.62607004e-34
        c = 299792458
        k = 1.380622e-23
        self.C_2 = h * c / k
        self.C_1 = 2 * np.pi * h * c * c
        self.C_0 = 7.0  # Constants for the absorptivity calculations

        self.base_path = base_path
        self.write_path = write_path
        self.tcp_soot = None
        self.dns_soot = None
        self.dns_optical_thickness = None
        self.save = save
        self.soot_error = None
        self.temperature_error = None
        self.dns_maximum_temperature = None
        self.dns_maximum_soot = None
        self.prf = None
        self.rhoC = 2000  # [kg / m^3]
        self.field_size = len(fields)
        if tcp_axis == "x":
            self.tcp_axis = 0
        if tcp_axis == "y":
            self.tcp_axis = 1
        if tcp_axis == "z":
            self.tcp_axis = 2

        # Initialize the
        vtcp = ChrestData(base_path + "/" + files)
        self.start_point = vtcp.start_point
        self.end_point = vtcp.end_point
        # self.rgb = np.zeros((np.shape(self.data)[0], np.shape(self.data)[1], self.field_size))
        self.times = np.zeros(self.field_size)
        self.names = np.zeros(self.field_size)
        self.tcp_temperature = None
        self.temperature_error = None
        # coords_tmp = vtcp.compute_cell_centers(3)
        # self.coords = np.zeros((self.field_size, np.shape(coords_tmp)[0], np.shape(coords_tmp)[1]))

        # Get the data from the vTCP files
        self.data = np.array([])
        for f in range(self.field_size):
            data_tmp, _, _ = vtcp.get_field(fields[f])
            if f == 0:
                self.data = np.expand_dims(data_tmp, axis=0)
            else:
                self.data = np.vstack((self.data, np.expand_dims(data_tmp, axis=0)))

        self.set_limits()  # Sets the time step range of the processing

    def get_tcp_temperature(self):
        # Calculate the two-color-pyrometry temperature of the frame
        # First, get the intensity ratio between the red and green channels (0 and 1)
        # Then, use the ratio to get the temperature
        # Finally, plot the temperature
        ratio = self.data[1, :, :, :, :] / self.data[0, :, :, :, :]
        ratio = np.nan_to_num(ratio)

        c = 3.e8  # Speed of light
        h = 6.626e-34  # Planck's constant
        k = 1.3806e-23  # Boltzmann Constant

        # Planck's first and second constant
        c1 = 2. * np.pi * h * c * c
        c2 = h * c / k

        lambdaR = 650e-9
        lambdaG = 532e-9
        self.tcp_temperature = np.zeros_like(ratio, dtype=np.dtype(float))

        threshold_fraction = 0.05  # Threshold for the absolute intensity (keep at 0.15?)

        for n in range(np.shape(self.data)[1]):
            for i in range(np.shape(self.data)[2]):
                for j in range(np.shape(self.data)[3]):
                    for k in range(np.shape(self.data)[4]):
                        if self.data[0, n, i, j, k] < threshold_fraction * np.max(self.data[0, n, :, :, :]) \
                                or self.data[1, n, i, j, k] < threshold_fraction * np.max(self.data[1, n, :, :, :]):
                            self.tcp_temperature[
                                n, i, j, k] = 0  # If either channel is zero, set the temperature to zero
                        if ratio[n, i, j, k] == 0:
                            self.tcp_temperature[n, i, j, k] = 0
                        else:
                            self.tcp_temperature[n, i, j, k] = (c2 * ((1. / lambdaR) - (1. / lambdaG))) / (
                                    np.log(ratio[n, i, j, k]) + np.log((lambdaG / lambdaR) ** 5))
                        if self.tcp_temperature[n, i, j, k] < 300:  # or self.tcp_temperature[i] > 3500:
                            self.tcp_temperature[n, i, j, k] = 300
        # return self.tcp_temperature

        # Get the size of a single mesh.
        # Iterate through the time steps
        # Iterate through each time step and place a point on the plot

    def get_optical_thickness(self, dns_data):
        # Calculate the optical thickness of the frame
        # First get the absorption for each cell in the dns
        dns_temperature, _, _ = dns_data.get_field("temperature")
        if self.dns_soot is None:
            self.get_dns_soot(dns_data)
        kappa = (3.72 * self.dns_soot * self.C_0 * dns_temperature) / self.C_2  # Soot mean absorption
        # Then sum the absorption through all cells in the ray line
        dns_sum_soot = kappa.sum(axis=(self.tcp_axis + 1), keepdims=True)
        self.dns_optical_thickness = dns_sum_soot * (
                self.end_point[self.tcp_axis] - self.start_point[self.tcp_axis])

    def get_tcp_soot(self):
        lambda_r = 650.e-9
        path_length = self.end_point[self.tcp_axis] - self.start_point[self.tcp_axis]
        self.tcp_soot = np.zeros_like(self.tcp_temperature)
        threshold_condition = (self.tcp_temperature > 400.0)
        self.tcp_soot = np.where(
            threshold_condition,
            (-lambda_r * 1.e9 / (self.C_0 * path_length)) * np.log(
                1 - (self.data[0, :, :, :, :] * (lambda_r ** 5) * np.exp(
                    self.C_2 / (lambda_r * self.tcp_temperature))) / self.C_1),
            self.tcp_soot
        )

    def get_dns_soot(self, dns_data):
        dns_density_yi, _, _ = dns_data.get_field("densityYi")
        self.dns_soot = dns_density_yi / self.rhoC

    def plot_temperature_step(self, n, name):
        # Get the tcp_temperature if it hasn't been computed already
        if self.tcp_temperature is None:
            self.get_tcp_temperature()  # Calculate the TCP temperature of the given boundary intensities

        tcp_temperature_frame = self.tcp_temperature[n, :, :, :]

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # plot the temperature as a slice in the z direction
        im = ax.imshow(tcp_temperature_frame[0, :, :],
                       interpolation='none', cmap="inferno",
                       origin='lower', extent=[self.start_point[0], self.end_point[0],
                                               self.start_point[1], self.end_point[1]],
                       vmax=4500, vmin=300)
        fig.colorbar(im, shrink=0.5, pad=0.05)
        # ax.clabel(CS, inline=True, fontsize=10)
        # ax.set_title('CHREST Format vTCP (n = ' + str(n) + ')')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(r"Temperature $[K]$")  # Add label for the temperature
        # if self.save:
        # plt.savefig(str(name) + "." + str(n).zfill(3) + ".png", dpi=1000, bbox_inchees='tight')
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
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if self.save:
            plt.savefig(self.write_path + "/" + (name) + "." + str(n).zfill(3) + ".png", dpi=1000, bbox_inches='tight')
        plt.show()

    def get_uncertainty_field(self, dns_data):
        if self.tcp_temperature is None:
            self.get_tcp_temperature()  # Calculate the TCP temperature of the given boundary intensities
        if self.tcp_soot is None:
            self.get_tcp_soot()  # Calculate the TCP temperature of the given boundary intensities

        # Now that we have the tcp temperature, we want to get the maximum temperatures in each of the ray lines.
        dns_temperature, _, _ = dns_data.get_field("temperature")
        dns_soot, _, _ = dns_data.get_field("densityYi")
        self.dns_maximum_temperature = dns_temperature.max(axis=(self.tcp_axis + 1), keepdims=True)
        self.dns_maximum_soot = dns_soot.max(axis=(self.tcp_axis + 1), keepdims=True)
        self.temperature_error = np.abs(self.dns_maximum_temperature - self.tcp_temperature)
        self.soot_error = np.abs(self.dns_maximum_soot - self.tcp_soot)

    def plot_optical_thickness(self, n):

        optical_thickness_frame = self.dns_optical_thickness[n, :, :, :]

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # plot the temperature as a slice in the z direction
        im = ax.imshow(optical_thickness_frame[0, :, :],
                       interpolation='none', cmap="inferno",
                       origin='lower', extent=[self.start_point[0], self.end_point[0],
                                               self.start_point[1], self.end_point[1]],
                       vmax=optical_thickness_frame.max(), vmin=optical_thickness_frame.min())
        fig.colorbar(im, shrink=0.5, pad=0.05, label=r"Optical Thickness")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.tight_layout()
        if self.save:
            plt.savefig(self.write_path + "/" + "opticalThickness" + "." + str(n).zfill(3) + ".png", dpi=1000,
                        bbox_inchees='tight')
        plt.show()

    def plot_uncertainty_field(self, n):
        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(3, 4, width_ratios=[20, 1, 20, 1], height_ratios=[1, 1, 1])

        tcp_temperature_frame = self.tcp_temperature[n, :, :, :]
        dns_temperature_frame = self.dns_maximum_temperature[n, :, :, :]
        temperature_error_frame = self.temperature_error[n, :, :, :]

        tcp_soot_frame = self.tcp_soot[n, :, :, :]
        dns_soot_frame = self.dns_maximum_soot[n, :, :, :]
        soot_error_frame = self.soot_error[n, :, :, :]

        # Plot dns_temperature
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(dns_temperature_frame[0, :, :], interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[0], self.end_point[0],
                                 self.start_point[1], self.end_point[1]],
                         vmax=4500, vmin=300)
        ax1.set_title("DNS Temperature")
        ax1.set_ylabel("y [m]")

        # Plot tcp_temperature
        ax2 = fig.add_subplot(gs[1, 0])
        im2 = ax2.imshow(tcp_temperature_frame[0, :, :],
                         interpolation='none', cmap="inferno",
                         origin='lower', extent=[self.start_point[0], self.end_point[0],
                                                 self.start_point[1], self.end_point[1]],
                         vmax=4500, vmin=300)
        ax2.set_title("TCP Temperature")
        ax2.set_ylabel("y [m]")

        # Plot uncertainty
        ax3 = fig.add_subplot(gs[2, 0])

        im3 = ax3.imshow(temperature_error_frame[0, :, :], interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[0], self.end_point[0],
                                 self.start_point[1], self.end_point[1]],
                         vmax=1000, vmin=0)
        ax3.set_title("Temperature Error Field")
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("y [m]")

        # Add colorbar
        cbar_ax1 = fig.add_subplot(gs[0:2, 1])
        cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', label="Temperature [K]")
        cbar_ax1.yaxis.set_ticks_position('right')
        cbar_ax1.yaxis.set_label_position('right')

        cbar_ax2 = fig.add_subplot(gs[2, 1])
        cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation='vertical', label="Error [K]")
        cbar_ax2.yaxis.set_ticks_position('right')
        cbar_ax2.yaxis.set_label_position('right')

        # Plot dns_soot
        ax4 = fig.add_subplot(gs[0, 2])
        im4 = ax4.imshow(dns_soot_frame[0, :, :], interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[0], self.end_point[0],
                                 self.start_point[1], self.end_point[1]],
                         vmax=tcp_soot_frame.max(), vmin=tcp_soot_frame.min())
        ax4.set_title("DNS Soot")
        ax4.set_ylabel("y [m]")

        # Plot tcp_soot
        ax5 = fig.add_subplot(gs[1, 2])
        im5 = ax5.imshow(tcp_soot_frame[0, :, :],
                         interpolation='none', cmap="inferno",
                         origin='lower', extent=[self.start_point[0], self.end_point[0],
                                                 self.start_point[1], self.end_point[1]],
                         vmax=tcp_soot_frame.max(), vmin=tcp_soot_frame.min())
        ax5.set_title("TCP Soot")
        ax5.set_ylabel("y [m]")

        # Plot uncertainty
        ax6 = fig.add_subplot(gs[2, 2])

        im6 = ax6.imshow(soot_error_frame[0, :, :], interpolation='none', cmap="inferno",
                         origin='lower',
                         extent=[self.start_point[0], self.end_point[0],
                                 self.start_point[1], self.end_point[1]],
                         vmax=soot_error_frame.max(), vmin=soot_error_frame.min())
        ax6.set_title("Soot Error Field")
        ax6.set_xlabel("x [m]")
        ax6.set_ylabel("y [m]")

        # Add colorbar
        cbar_ax4 = fig.add_subplot(gs[0:2, 3])
        cbar3 = fig.colorbar(im4, cax=cbar_ax4, orientation='vertical', label="Soot Volume Fraction")
        cbar_ax4.yaxis.set_ticks_position('right')
        cbar_ax4.yaxis.set_label_position('right')

        cbar_ax5 = fig.add_subplot(gs[2, 3])
        cbar4 = fig.colorbar(im6, cax=cbar_ax5, orientation='vertical', label="Soot Error")
        cbar_ax5.yaxis.set_ticks_position('right')
        cbar_ax5.yaxis.set_label_position('right')

        plt.tight_layout()
        if self.save:
            plt.savefig(self.write_path + "/" + "uncertaintyField" + "." + str(n).zfill(3) + ".png", dpi=1000,
                        bbox_inches='tight')
        plt.show()

    def plot_line_of_sight(self, n, data_3d):
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))

        # Create an array for the x-axis.
        x = data_3d.grid
        x = np.transpose(x, (1, 2, 0))

        dns_temperature, _, _ = data_3d.get_field("temperature")

        # Plot temperature along the line of sight
        ax[0].scatter(x, dns_temperature[n, :, :, :], color='r', label='Temperature')
        ax[0].set_title("Temperature along the line of sight")
        ax[0].set_ylabel("Temperature [K]")
        ax[0].legend()

        # Plot soot volume fraction along the line of sight
        ax[1].scatter(x, self.dns_soot[n, :, :, :], color='b', label='Soot Volume Fraction')
        ax[1].set_title("Soot volume fraction along the line of sight")
        ax[1].set_ylabel("Soot Volume Fraction")
        ax[1].legend()

        # Calculate the absorption for each cell in the line of sight
        kappa = (3.72 * self.dns_soot[n, :, :, :] * self.C_0 * self.dns_temperature[n, :, :, :]) / self.C_2

        # Plot absorption coefficient along the line of sight
        ax[2].scatter(x, kappa, color='g', label='Absorption Coefficient')
        ax[2].set_title("Absorption coefficient along the line of sight")
        ax[2].set_xlabel("Position along the line of sight")
        ax[2].set_ylabel("Absorption Coefficient")
        ax[2].legend()

        plt.tight_layout()
        if self.save:
            plt.savefig(self.write_path + "/" + "line_of_sight" + ".png", dpi=1000, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate plots from virtual TCP data. '
                    'See https://github.com/cet-lab/experimental-post-processing/wiki'
                    '/Matlab-To-XdmfGenerator for details.  ')
    parser.add_argument('--base_path', dest='base_path', type=str, required=True,
                        help='The path to the ablate hdf5 file(s) containing the ablate data.'
                             '  A wild card can be used '
                             'to supply more than one file.')
    parser.add_argument('--file', dest='hdf5_file', type=str, required=True,
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
                        help='Impacts the saturation of the virtual camera.', required=False)
    parser.add_argument('--name', dest='name', type=str,
                        help='What to call the outputs.', required=True)
    parser.add_argument('--dns', dest='dns', type=str,
                        help='Path to the volumetric DNS data.', required=True)
    parser.add_argument('--tcp_axis', dest='tcp_axis', type=str,
                        help='Direction of the tcp axis.', required=True)
    parser.add_argument('--write_path', dest='write_path', type=str, required=False,
                        help='Path to save figures to.')
    parser.add_argument('--save', dest='save', type=bool, required=False,
                        help='Event names to measure.')

    args = parser.parse_args()
    if args.deltaT is None:
        args.deltaT = 0.004
    if args.write_path is not None:
        write_path = args.write_path
    else:
        write_path = args.base_path + "/figures"
    if args.save is None:
        args.save = True
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    vTCP = VTcpData(args.hdf5_file, args.fields, args.tcp_axis, args.base_path, write_path, args.save)  # Initialize
    # the virtual TCP creation.

    print(len(vTCP.data[0, :, 0]))

    # Get the CHREST data associated with the simulation for the 3D stuff
    data_3d = ChrestData(args.base_path + "/" + args.dns)
    vTCP.get_uncertainty_field(data_3d)
    vTCP.get_optical_thickness(data_3d)

    # Calculate the difference between the DNS temperature and the tcp temperature
    # for i in range(np.shape(vTCP.data)[1]):
    #     vTCP.plot_optical_thickness(i)
    # for i in range(np.shape(vTCP.data)[1]):
    #     vTCP.plot_uncertainty_field(i)

    vTCP.plot_line_of_sight(50, data_3d)

    # It would be worth correlating the uncertainty field to something non-dimensional
    # Or at least related to the flame structure so that it can be generalized
    # Maybe the mixture fraction is an appropriate value?

    print('Done')
