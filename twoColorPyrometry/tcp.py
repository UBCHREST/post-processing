import numpy as np  # Basic manipulation
from numba import jit, cuda  # Enables gpu programming for cuda devices
import multiprocessing as mproc  # Multiprocess CPU programming.


# %% In the main execution:

def main():
    print("Hello World!")


# Hardware init: Check for gpu, check for number of available processes. Spawn the appropriate process for hardware.
# Spawn a hardware instance containing the functions which work on the images

# Make necessary directories for writing files.

# Create an object for the image containing all the necessary fields for the image processing output.
# Load in an image (The image file could be part of the image object constructor)
# Set geometric information. (Belongs to the subclass properties)

# Pass the image object to the hardware object to do the work on it. (Process and save)

if __name__ == "__main__":
    main()


# %% Hardware object (Depending on the type of hardware (serial, parallel (nproc), gpu) different functions are called)

class Hardware:

    def __init__(self):
        self.data = []


# %% Image object or struct (Contains all the field information about the images that are being written)
# Multiple subclasses might include different dimensions or coordinate systems.

class Image:

    def __init__(self):
        self.data = []
