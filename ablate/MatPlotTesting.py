import matplotlib.pyplot as plt
import argparse
import pathlib
import numpy
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Loads in the hdf5 file location")
    parser.add_argument("--file", dest="file", type=pathlib.Path, required=True, help="Loads hdf5 file")
    parser.add_argument("--save", dest="save", type=bool, required=False, default=False,
                        help="Decide whether to save the file or show it")
    args = parser.parse_args()

    # Load hdf5 file in
    hdf5 = h5py.File(args.file, "r")

    # Load in all datasets
    main = hdf5["main"]
    avg = main["temperatureRMS_avg"]
    maximum = main["temperatureRMS_max"]
    minimum = main["temperatureRMS_min"]

    # Load in discretization + start
    dx = main["discretization"][0]
    dy = main["discretization"][1]
    start = main["start"]

    # Assemble corners
    xLen = avg.shape[1]
    yLen = avg.shape[0]
    xEnd = start[1] + (xLen+1) * dx
    yEnd = start[0] + (yLen+1) * dy
    xRange = numpy.arange(start[1], xEnd, dx)
    yRange = numpy.arange(start[0], yEnd, dy)
    X, Y = numpy.meshgrid(xRange, yRange)

    fig, ax = plt.subplots(layout='tight')
    thePlot = ax.pcolormesh(X, Y, avg)
    ax.axis('equal')
    ax.set_xlim(left=xRange[0], right=xRange[-1])
    fig.colorbar(thePlot, location='bottom', orientation='horizontal')

    # Output figure
    if args.save:
        output_image = args.file.parent / (args.file.name + ".jpg")
        plt.savefig(output_image)
    else:
        plt.show()
