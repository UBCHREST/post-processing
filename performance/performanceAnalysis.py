import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting
from os.path import exists
from scipy.optimize import curve_fit
import argparse
import os
import scipy
from scipy import ndimage


# Template path: "outputs/Scaling2D_30_16_[105, 15].xml"
# basePath = "slabRadSF2DScaling/scalingCsv/volumetricCsv/"
# basePath = "csvFiles/"
# initName = b"Radiation::Initialize"
# solveName = b"Radiation::EvaluateGains"

class PerformanceAnalysis:

    def __init__(self, base_path=None, name=None, processes=None, faces=None, cell_size=None, rays=None, events=None,
                 write_path=None):
        self.times = None
        self.markerarray = None
        self.colorarray = None
        self.base_path = base_path
        self.write_path = write_path
        self.name = name
        self.processes = processes
        self.faces = faces
        self.cell_size = cell_size
        self.events = np.asarray(events, dtype=bytes)
        self.rays = rays
        self.set_plot_parameters()
        self.processes_mesh, self.problems_mesh = np.meshgrid(self.processes, self.cell_size)

    @staticmethod
    def r_squared_func(y, y_fit):
        return 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Do curve fitting of the data for performance modelling purposes.
    @staticmethod
    def ray_recombination_function(d, p, c, r, alpha, beta, f, g):
        return r * ((c / p) ** ((d + 1) / d)) * f + alpha * np.log2(p) + r * (c ** ((d - 1) / d)) * (
                p ** (1 / d)) * 2 * beta + r * ((c / p) ** ((d - 1) / d)) * (p ** (1 / d)) * g

    @staticmethod
    def segment_evaluation_function(d, p, c, r, alpha, beta, f, g):
        d = 1
        return r * ((c / p) ** ((d + 1) / d)) * f

    @staticmethod
    def communication_function(d, p, c, r, alpha, beta, f, g):
        return alpha * np.log2(p) + r * (c ** ((d - 1) / d)) * (
                p ** (1 / d)) * 2 * beta

    @staticmethod
    def ray_collapse_function(d, p, c, r, alpha, beta, f, g):
        return r * ((c / p) ** ((d - 1) / d)) * (p ** (1 / d)) * g

    # For communication cost models:
    # beta : bandwidth cost of the network
    # n : size of the data
    # p : number of processes
    # alpha : latency cost of the network
    # gamma : cost of computation

    @staticmethod
    def broadcast(p, n, alpha, beta):
        return np.log2(p) * (3.0 * alpha + n * beta)

    @staticmethod
    def reduce(p, n, alpha, beta, gamma):  # Additional cost associated with computation of reduction.
        return np.log2(p) * (3.0 * alpha + n * beta + n * gamma)

    @staticmethod
    def scatter(p, n, alpha, k, beta):  # And gather
        return 3.0 * alpha * np.log2(p) + ((p - 1.0) * n * beta) / p

    @staticmethod
    def gather(p, n, alpha, k, beta):  # And gather
        return 3.0 * alpha * np.log2(p) + ((p - 1.0) * n * beta) / p

    @staticmethod
    def bidirectional_all_gather(p, n, alpha, beta):
        return 3.0 * alpha * np.log2(p) + (n * beta)

    @staticmethod
    def bidirectional_reduce_scatter(p, n, alpha, beta, gamma):
        return np.log2(p) * 3.0 * alpha + ((p - 1.0) / p) * n * (beta + gamma)

    @staticmethod
    def bucket_all_gather(p, n, alpha, beta):
        return p * alpha + ((p - 1.0) / p) * n * beta

    @staticmethod
    def bucket_reduce_scatter(p, n, alpha, beta, gamma):
        return p * alpha + ((p - 1.0) / p) * n * (beta + gamma)

    @staticmethod
    def all_reduce(p, n, alpha, beta, gamma):
        return np.log2(p) * (3.0 * alpha + n * beta + n * gamma)

    def set_plot_parameters(self):
        # Set up plotting options that must be defined by the user
        self.colorarray = ["o", "s", "o", "s", ".", ".", ".", ".",
                           ".", ".", ".", ".", ".", ".",
                           "."]
        self.markerarray = [".", "1", "P", "*"]
        plt.rcParams["font.family"] = "Noto Serif CJK JP"

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f(self.markerarray[i], "k") for i in range(len(self.markerarray))]
        handles += [f("s", "black") for i in range(len(self.colorarray))]

    def load_csv_files(self):
        # Create arrays which the parsed information will be stored inside: Whatever information is desired
        self.times = np.zeros([len(self.events), len(self.rays), len(self.processes), len(self.faces)])
        # Iterate through the arrays to get information out of the files
        for r in range(len(self.rays)):
            for p in range(len(self.processes)):
                for f in range(len(self.faces)):
                    # Create strings which represent the file names of the outputs
                    path = self.base_path + "/" + self.name + "_" + str(self.rays[r]) + "_" + str(
                        self.processes[p]) + "_" + str(
                        self.faces[f]) + ".csv"  # File path
                    dtypes = {'names': ('stage', 'name', 'time'),
                              'formats': ('S30', 'S30', 'f4')}

                    if exists(path):  # If the path exists then it can be loaded
                        data = np.loadtxt(path, delimiter=",", dtype=dtypes, skiprows=1, usecols=(0, 1, 4))
                        lines = len(data)  # Get the length of the csv
                        for e in range(len(self.events)):

                            for i in range(lines):  # Iterate through all the lines in the csv
                                if (data[i][1] == self.events[e]) and (
                                        data[i][2] > self.times[e, r, p, f]):  # Check current line
                                    self.times[e, r, p, f] = data[i][
                                        2]  # If it is, then write the value in column 4 of that line

                            # If the time is never written then the filter code then don't try to plot it
                            if self.times[e, r, p, f] == 0:
                                self.times[e, r, p, f] = float("nan")

        # Calculate the sum over the 'e' dimension
        sum_over_e = np.sum(self.times, axis=0)

        # Expand the dimensions of sum_over_e to match self.times
        # Here we assume that self.times has 4 dimensions, and 'e' corresponds to the 0th dimension
        sum_over_e = sum_over_e[np.newaxis, :, :, :]

        # Concatenate sum_over_e with self.times along the 'e' dimension
        self.times = np.concatenate((self.times, sum_over_e), axis=0)

        self.processes = np.asarray(self.processes)
        self.faces = np.asarray(self.faces)
        self.rays = np.asarray(self.rays)
        self.events = np.append(self.events, "Total")

    # Set bounds for the parameters (all non-negative)
    # param_bounds = ([0, -np.inf, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
    param_bounds = ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])

    def plot_static_scaling(self):
        for e in range(len(self.events)):
            # Initialization static scaling analysis
            plt.figure(figsize=(10, 6), num=1)
            for n in range(len(rays)):
                for i in range(len(self.processes)):
                    mask = np.isfinite(self.times[e, n, i, :])
                    x = self.cell_size
                    y = self.cell_size / self.times[e, n, i, :]
                    plt.loglog(x[mask], y[mask], linewidth=1, marker=self.markerarray[n], c=self.colorarray[i])
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            plt.xlabel(r'DOF $[cells]$', fontsize=10)
            plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
            labels = 0
            labels = np.append(labels, self.processes)
            plt.legend(self.handles, labels, loc="upper left")
            if not os.path.exists(self.write_path + "/figures"):
                os.makedirs(self.write_path + "/figures")
            plt.savefig(self.write_path + "/figures/" + self.name + str(self.events[e]) + '_static_scaling.png',
                        dpi=1500, bbox_inches='tight')
            plt.show()

    def plot_weak_scaling(self, discretization_index, cells_per_process):
        r = discretization_index
        for e in range(len(self.events)):
            # Coordinates of the line we'd like to sample along
            line = [(0, 0),
                    (self.processes.max() * cells_per_process, self.processes.max())]

            # Convert the line to piself.problems_meshel/index coordinates
            x_world, y_world = np.array(list(zip(*line)))
            col = self.times[e, r, :, :].shape[1] * (x_world - self.problems_mesh.min()) / self.problems_mesh.ptp()
            row = self.times[e, r, :, :].shape[0] * (y_world - self.processes_mesh.min()) / self.processes_mesh.ptp()

            # Interpolate the line at "num" points...
            num = 100
            row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

            # Extract the values along the line, using cubic interpolation
            zi = ndimage.map_coordinates(self.times[e, r, :, :], np.vstack((row, col)))

            # Plot...
            fig, axes = plt.subplots(nrows=2, figsize=(6, 6))
            axes[0].pcolormesh(self.problems_mesh, self.processes_mesh, np.transpose(self.times[e, r, :, :]))
            axes[0].plot(x_world, y_world, 'ro-')
            axes[0].axis('image')
            axes[0].set_box_aspect(1)

            # axes[0].set_aspect('equal')

            # axes[1].plot(x_world, zi)

            axes[1].plot(zi)
            plt.show()

    def plot_strong_scaling(self, function_fit):
        # Initialization Strong scaling analysis
        plt.figure(figsize=(6, 4), num=4)
        # plt.title("Solve Strong Scaling" + dims, pad=1)
        for e in range(len(self.events)):
            for n in range(len(rays)):
                for i in range(len(self.faces)):
                    mask = np.isfinite(self.times[e, n, :, i])
                    x = self.processes
                    y = self.times[e, n, :, i]

                    # Bring the lowest available index to the line to normalize the scaling plot *
                    # (ideal / lowest available index)
                    first = np.argmax(mask)

                    if np.sum(mask) > 5 and function_fit is not None:
                        # Perform non-linear curve fitting
                        popt, pcov = curve_fit(function_fit, x[mask], y[mask])  # , bounds=param_bounds)

                        # Extract the fitted parameters
                        t0, s, c, d, f = popt

                        # Calculate the R-squared value
                        y_fit = function_fit(x[mask], t0, s, c, d, f)
                        r_squared = self.r_squared_func(y[mask], y_fit)

                        plt.loglog(x[mask], (self.processes[first] * y[first]) / y_fit, c="black", linestyle="-.",
                                   label=f'Fitted curve: t0={t0:.2f}, s={s:.2f}, c={c:.2f}, r^2={r_squared:.2f}')
                        # print(f't0={t0:.2f}, s={s:.2f}, c={c:.2f}, d={d:.2f}, f={f:.2f}, r^2={r_squared:.2f}')
                    adjusted = (self.processes[first] * y[first]) / y[mask]
                    plt.loglog(x[mask], adjusted, linewidth=1, marker=self.colorarray[i],
                               c="black", markersize=4)
            plt.plot(self.processes, self.processes, linewidth=1, c="black", linestyle="--")
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            plt.xlabel(r'MPI Processes', fontsize=10)
            plt.ylabel(r'Speedup', fontsize=10)
            # labels = dtheta
            # labels = np.append(labels, faces)
            # plt.legend(["2D"], loc="upper left")  # , "3D"
            # plt.legend()
            if not os.path.exists(self.write_path + "/figures"):
                os.makedirs(self.write_path + "/figures")
            plt.savefig(self.write_path + "/figures/" + self.name + str(self.events[e]) + '_strong_scaling.png',
                        dpi=1500, bbox_inches='tight')
            plt.show()

    def plot_performance_contour(self, discretization_index):
        # Initialization Strong scaling analysis
        plt.figure(figsize=(6, 4), num=4)
        r = discretization_index
        for e in range(len(self.events)):
            # performance = np.zeros_like(self.times[e, r, :, :])
            # for p in range(len(self.processes)):
            #     performance[p, :] = self.cell_size / self.times[e, r, p, :]

            # Calculate performance
            # performance = cellsize_mesh / problems_mesh
            # Set log scale for contour levels
            min_time = np.min(self.times[e, r, :, :][self.times[e, r, :, :] > 0])
            max_time = np.max(self.times[e, r, :, :])
            num_levels = 5
            levels = np.logspace(np.log10(min_time), np.log10(max_time), num_levels)

            contour = plt.contour(self.problems_mesh, self.processes_mesh, np.transpose(self.times[e, r, :, :]),
                                  levels=levels)

            # Add contour labels
            plt.clabel(contour, inline=True, fontsize=8, fmt='%1.2f')

            # Set axis labels and ticks
            plt.ylabel('MPI Processes', fontsize=10)
            plt.xlabel('DOF', fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # Set log scale for both axes
            ax = plt.gca()
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Save the figure and display the plot
            if not os.path.exists(self.write_path + "/figures"):
                os.makedirs(self.write_path + "/figures")
            plt.savefig(self.write_path + "/figures/" + self.name + str(self.events[e]) + '_performance_contour.png',
                        dpi=1500, bbox_inches='tight')
            plt.show()

    def plot_time(self):
        # Initialization Strong scaling analysis
        plt.figure(figsize=(6, 4), num=4)
        # Constant parameters
        d = 1
        c = 50000
        r = 30

        for e in range(len(self.events)):
            for n in range(len(self.rays)):
                for i in range(len(self.faces)):
                    mask = np.isfinite(self.times[e, n, :, i])
                    x = self.processes[mask]
                    y = self.times[e, n, mask, i]

                    # define the initial parameter guess for alpha, beta, f, g
                    init_guess = [1, 1, 1, 1]

                    # use curve_fit function to fit data
                    # the function ray_recombination_function should have two arguments: the x values and the parameters to be fitted
                    params_opt, params_cov = curve_fit(
                        lambda x, alpha, beta, f, g: self.ray_recombination_function(d, x, c, r, alpha, beta, f, g),
                        x, y, p0=init_guess)

                    # generate y-values based on the fit
                    y_fit = self.ray_recombination_function(d, x, c, r, *params_opt)

                    plt.loglog(x, y, linewidth=1, marker=self.colorarray[i], c="black", markersize=4)
                    # plot fitted curve with a different color or line style
                    plt.loglog(x, y_fit, '--')

        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.xlabel(r'MPI Processes', fontsize=10)
        plt.ylabel(r'Speedup', fontsize=10)

        if not os.path.exists(self.write_path + "/figures"):
            os.makedirs(self.write_path + "/figures")

        plt.savefig(self.write_path + "/figures/" + self.name + str(self.events[e]) + '_times.png',
                    dpi=1500, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process scaling data from PETSc events.')
    parser.add_argument('--path', dest='base_path', type=str, required=True,
                        help='Path to the base directory containing scaling csv files.')
    parser.add_argument('--name', dest='name', type=str, required=True,
                        help='Title of the files being processed.')
    parser.add_argument('--processes', dest='processes', type=int, required=True, nargs="+",
                        help='Process counts being considered.')
    parser.add_argument('--problems', dest='problems', type=str, required=True, nargs="+",
                        help='Problems being considered.')
    parser.add_argument('--dof', dest='cell_size', type=float, required=True, nargs="+",
                        help='Mesh or dof size associated with each problem.')
    parser.add_argument('--events', dest='events', type=str, required=True, nargs="+",
                        help='Event names to measure.')
    parser.add_argument('--write_to', dest='write_path', type=str, required=False,
                        help='Event names to measure.')

    args = parser.parse_args()
    rays = np.array([15])

    if args.write_path is not None:
        write_path = args.write_path
    else:
        write_path = args.base_path

    scaling_data = PerformanceAnalysis(args.base_path, args.name, args.processes,
                                       args.problems, args.cell_size, rays, args.events, write_path)
    scaling_data.load_csv_files()
    # scaling_data.plot_performance_contour(0)
    # scaling_data.plot_strong_scaling(None)
    # scaling_data.plot_weak_scaling(0, 20.0)
    # scaling_data.plot_static_scaling()
    scaling_data.plot_time()

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # plt.rcParams["font.family"] = "Noto Serif CJK JP"
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # Define the parameters
    # alpha = [10.0E-6, 2.0E-6]  # Communication latency
    # beta = [1/600E9, 1/21.E9]  # Memory bandwidth cost
    # linestyles = ['-', '--', '-.']
    # d = 3  # Dimensions
    # p = np.logspace(1, 7, 400)  # MPI processes
    #
    # plt.figure(figsize=(8, 6))
    # for i in range(len(alpha)):
    #     # Compute the discretization N
    #     N = alpha[i] * np.log2(p) * (p ** (1 / d) - 1) / (p ** (1 / d) * (7 * beta[i]) - 5 * beta[i])
    #
    #     # Create the plot
    #     plt.loglog(p, N, color='k', linestyle=linestyles[i])
    # # plt.fill_between(p, 0, N, alpha=0, color='black')  # Fill the area under the curve
    #
    # # # Add horizontal lines at y=1E6 and y=1E4
    # # plt.axhline(y=1E6, color='k', linestyle='--')
    # # plt.axhline(y=1E4, color='k', linestyle='--')
    #
    # # Add labels and title
    # plt.xlabel('MPI Processes', fontsize=12)
    # plt.ylabel('N', fontsize=12)
    # # plt.ylim([0, 1E7])
    #
    # # Add annotations
    # # plt.annotate('Sweeping Method', xy=(0.25, 0.75), xycoords='axes fraction', ha='center', color='black')
    # # plt.annotate('Ray Recombination Method', xy=(0.75, 0.25), xycoords='axes fraction', ha='center', color='black')
    #
    # plt.legend(["Nvidia GPU Cluster", "Intel Xeon Cluster", "CPU2"], frameon=False)
    # plt.grid(True)
    # savePath = "/home/owen/CLionProjects/ParallelRadiationJCP/figures/"
    # plt.savefig(savePath + 'PerformanceTradeoffPoint', dpi=1000, bbox_inches='tight')
    # plt.show()

# --path /home/owen/ablateInputs/ScalingTests/csvFiles --name volumetricSFScaling --processes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 --problems [105,15] [149,21] [297,42] --dof 1575 3129 12474 --events Radiation::Initialize Radiation::EvaluateGains --write_to /home/owen/CLionProjects/ParallelRadiationJCP
# --path /home/owen/1d_scaling --name irradiation --processes 36 72 144 288 576 1152 2304 --problems [50000] --dof 50000 --events Radiation::EvaluateGains::Communication Radiation::EvaluateGains::Recombination Radiation::EvaluateGains::LocalSegmentIntegration --write_to /home/owen/CLionProjects/ParallelRadiationJCP
