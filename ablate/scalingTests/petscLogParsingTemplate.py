import xml.etree.ElementTree as ET  # xml parsing library
import numpy as np  # for matrix manipulation
import matplotlib.pyplot as plt  # for plotting
from os.path import exists  # don't try to open files that aren't there

# STEP 1: PASTE tmp DIRECTORIES INTO THE outputs FOLDER
# STEP 2: RUN getXml.bash TO EXTRACT PETSC LOG VIEW FILES
# STEP 3: RUN petscLogParsingTemplate WITH SPECIFIC OPTIONS AND NAMING TO PLOT SCALING DATA

# Name of the event that is being tracked
eventName = "Name that was given to the event in petsc log view"
# Template path: "outputs/Scaling2D_16_[105,15].xml"
# PetscLog output files should be procedurally named using the "enumerator" library.
basePath = "outputs/Scaling"

# Define the arrays that contain options which were used. These form the axes and strings for path locations.
processes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
faces = ["[105,15]", "[149,21]", "[210,30]", "[297,42]", "[420,60]", "[594,85]", "[840,120]"]
cellcount = [105 * 15, 149 * 21, 210 * 30, 297 * 42, 420 * 60, 594 * 85, 840 * 120]
dims = " 2D"

# Set up options that must be defined by the user
colorarray = ["black", "grey", "firebrick", "lightcoral", "darkorange", "goldenrod", "yellow", "yellowgreen",
              "green", "lightgreen", "teal", "powderblue", "darkorchid", "violet",
              "palevioletred"]

# Create arrays which the parsed information will be stored inside: Whatever information is desired
time = np.zeros((len(processes), len(faces)))

# Iterate through the arrays to get information out of the xml files
for p in range(len(processes)):
    for f in range(len(faces)):
        # Create strings which represent the file names of the outputs
        # IF USING AN ALTERNATIVE PATH NAMING SCHEME: CHANGE THE PROCEDURE HERE
        path = basePath + dims + "_" + str(processes[p]) + "_" + str(faces[f]) + ".xml"  # File path
        if exists(path):  # Make sure not to try accessing a path that doesn't exist
            tree = ET.parse(path)  # Create element tree objects
            root = tree.getroot()  # Get root element
            # Iterate items (the type of event that contains the data)
            for item in root.findall('./petscroot/selftimertable/event'):
                # Get the specific name of the event that is desired
                #    Get the sub-value that is desired out of the event
                if item.find("name").text == eventName:
                    if not item.find('time/avgvalue') is None:
                        time[p, f] = item.find('time/avgvalue').text
        # Don't plot values of zero on a log log plot
        if time[p, f] == 0:
            time[p, f] = float("nan")

processes = np.asarray(processes)
faces = np.asarray(faces)

# Static Scaling Analysis
plt.figure(figsize=(10, 6), num=1)
plt.title("Static Scaling " + dims, pad=1)
for i in range(len(processes)):
    mask = np.isfinite(time[i, :])
    x = cellcount[:]
    y = cellcount[:] / time[i, :]
    plt.loglog(x[mask], y[mask], linewidth=1, c=colorarray[i])
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel(r'DOF $[cells]$', fontsize=10)
plt.ylabel(r'Performance $[\frac{DOF}{s}]$', fontsize=10)
plt.legend(colorarray, loc="upper left")
plt.savefig('StaticScaling' + dims, dpi=1500, bbox_inches='tight')
plt.show()

# Initialization Strong Scaling Analysis
plt.figure(figsize=(10, 6), num=2)
plt.title("Strong Scaling " + dims, pad=1)
for i in range(len(faces)):
    mask = np.isfinite(time[:, i])
    x = processes
    y = time[:, i]
    # Bring the lowest available index to the line to normalize the scaling plot * (ideal / lowest available index)
    first = np.argmax(mask)
    # Get the first
    plt.loglog(x[mask], (processes[first] * y[first]) / y[mask], linewidth=1, c=colorarray[i])
plt.plot(processes, processes, linewidth=1, c="black", linestyle="--")
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel(r'Processes', fontsize=10)
plt.ylabel(r'Speedup', fontsize=10)
plt.legend(colorarray, loc="upper left")
plt.savefig('StrongScaling' + dims, dpi=1500, bbox_inches='tight')
plt.show()
