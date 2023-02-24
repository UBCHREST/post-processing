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

vtcp = ablateData.AblateData(
    "/home/owen/ablateInputs/slabBurner.pmma.3D/_slabBurner.pmma.3D.22G.virtualTCP_2023-02-23T21-06-55/boundaryFacesFront_radiationFluxMonitor/boundaryFacesFront_radiationFluxMonitor.*.hdf5")
# vtcp = ablateData.AblateData(
#     "/home/owen/ablateInputs/slabBurner.pmma.3D/_slabBurner.pmma.3D.22G.virtualTCP_2023-02-23T21-06-55/domain/domain.00000.hdf5")
# vtcp.create_field("colors", 3, ["red", "green", "blue"])
[data, times, names] = vtcp.get_field("boundaryFacesFront_radiationFluxMonitor_tcpTracer0")
coords = vtcp.compute_cell_centers(3)

# Get the size of a single mesh.
# Iterate through the time steps
# Iterate through each time step and place a point on the plot
n = 40

# frame = [coords[:, 0], coords[:, 1], data[n, :]]
frame = np.vstack((coords[:, 0], coords[:, 1], data[n, :]))
frame = np.transpose(frame)

d = pd.DataFrame(frame, columns=['x', 'y', 'd'])

D = d.pivot_table(index='x', columns='y', values='d').T.values

X_unique = np.sort(d.x.unique())
Y_unique = np.sort(d.y.unique())
X, Y = np.meshgrid(X_unique, Y_unique)

# x = coords[:, 0]
# y = coords[:, 1]
# d = data[n, :]
# X, Y = np.meshgrid(x, y)
# # D = np.griddata(d)

fig, ax = plt.subplots()
ax.set_aspect('equal')
CS = ax.imshow(D, interpolation='bilinear', cmap="inferno",
               origin='lower', extent=[frame[:, 0].min(), frame[:, 0].max(), frame[:, 1].min(), frame[:, 1].max()],
               vmax=abs(D).max(), vmin=-abs(D).max())
# ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('CHREST Format vTCP')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
plt.savefig('vTCP_test', dpi=1000, bbox_inches='tight')
plt.show()

# Save mp4 out of all the frames stiched together.
print('Done')
