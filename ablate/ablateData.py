import argparse
import pathlib
import numpy as np
import h5py

from chrest.chrestData import ChrestData
from support.supportPaths import expand_path
from scipy.spatial import KDTree


class AblateData:
    # a list of ablate files
    files = None

    # store the files based upon time
    files_per_time = dict()
    times = []

    # store the cells and vertices
    cells = None
    vertices = None

    # load the metadata from the first file
    metadata = dict()

    """
    Creates a new class from hdf5 chrest formatted file(s).
    """

    def __init__(self, files=None):
        if files is None:
            self.files = []
        elif isinstance(files, str):
            self.files = expand_path(files)
        elif isinstance(files, pathlib.Path):
            self.files = expand_path(files)
        else:
            self.files = files

        # Open each file to get the time and check the available fields
        for file in self.files:
            # Load in the hdf5 file
            hdf5 = h5py.File(file, 'r')

            # If not set, copy over the cells and vertices
            if self.cells is None:
                self.cells = hdf5["viz"]["topology"]["cells"]
                self.vertices = hdf5["geometry"]["vertices"]

            # extract the time
            time = hdf5['time'][0][0]
            self.times.append(time)

            self.files_per_time[time] = file

        # Store the list of times
        self.times = list(self.files_per_time.keys())
        self.times.sort()

    """
    computes the cell center for each cell [c, d]
    """

    def compute_cell_centers(self, dimensions):
        # create a new np array based upon the dim
        number_cells = self.cells.shape[0]

        vertices = self.vertices[:]
        vertices_dim = vertices.shape[1]

        coords = np.zeros((number_cells, dimensions))

        # march over each cell
        for c in range(len(coords)):
            cell_vertices = vertices.take(self.cells[c], axis=0)

            # take the average
            cell_center = np.sum(cell_vertices, axis=0)
            cell_center = cell_center / len(cell_vertices)

            # put back
            coords[c, 0:vertices_dim] = cell_center

        return coords

    """
    gets the specified field and the number of components
    """

    def get_field(self, field_name):
        # create a dictionary of times/data
        data = []
        components = 0
        # march over the files
        for t in self.times:
            # Load in the file
            with h5py.File(self.files_per_time[t], 'r') as hdf5:
                try:
                    # Check each of the cell fields
                    hdf5_field = hdf5['cell_fields'][field_name]
                    # load in the specified field name
                    hdf_field_data = hdf5_field[0, :]
                    data.append(hdf_field_data)

                    if len(hdf_field_data.shape) > 1:
                        components = hdf_field_data.shape[1]
                except Exception as e:
                    raise Exception(
                        "Unable to open field " + field_name + "." + str(e))

        return np.stack(data), components

    """
    converts the supplied fields ablate object to chrest data object
    """

    def map_to_chrest_data(self, chrest_data, field_mapping):
        # get the cell centers for this mesh
        cell_centers = self.compute_cell_centers(chrest_data.dimensions)

        # add all the ablate times to chrest
        chrest_data.times = self.times.copy()

        # store the fields in ablate we need
        ablate_fields = list(field_mapping.keys())

        # store the chrest data in the same order
        ablate_field_data = []
        chrest_field_data = []
        components = []

        # create the new field in the chrest data
        for ablate_field in ablate_fields:
            chrest_field = field_mapping[ablate_field]
            ablate_field_data_tmp, components_tmp  = self.get_field(ablate_field)

            ablate_field_data.append(ablate_field_data_tmp)
            components.append(components_tmp)
            chrest_field_data.append(chrest_data.create_field(chrest_field, components_tmp)[0])


        # build a list of k, j, i points to iterate over
        chrest_coords = chrest_data.get_coordinates()

        # size up the storage value
        chrest_cell_number = np.prod(chrest_data.grid)

        # reshape to get a single list order back
        chrest_cell_centers = chrest_coords.reshape((chrest_cell_number, chrest_data.dimensions))

        # now search and copy over data
        tree = KDTree(cell_centers)
        _, points = tree.query(chrest_cell_centers)

        # march over each field
        for f in range(len(ablate_field_data)):
            # get in the correct order
            ablate_field_in_chrest_order = ablate_field_data[f][:, points]

            # reshape it back to k,j,i
            ablate_field_in_chrest_order = ablate_field_in_chrest_order.reshape(
                chrest_field_data[f].shape
            )

            # copy over the data
            chrest_field_data[f][:] = ablate_field_in_chrest_order[:]


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a chrest data file from an ablate file')
    parser.add_argument('--file', dest='file', type=pathlib.Path, required=True,
                        help='The path to the ablate hdf5 file(s) containing the ablate data.'
                             '  A wild card can be used '
                             'to supply more than one file.')
    args = parser.parse_args()

    # this is some example code for chest file post-processing
    ablate_data = AblateData(args.file)

    # list the fields to map
    field_mappings = {"aux_temperature": "temperature", "aux_velocity": "vel"}

    # create a chrest data
    chrest_data = ChrestData()
    chrest_data.setup_new_grid([0., 0.0, -0.0127], [0.1, 0.0254, 0.0127], [0.001, 0.001, 0.001])
    # map the ablate data to chrest
    ablate_data.map_to_chrest_data(chrest_data, field_mappings)

    # write the new file without wild card
    chrest_data_path_base = args.file.parent / (str(args.file.stem).replace("*", "") + ".chrest")
    chrest_data_path_base.mkdir(parents=True, exist_ok=True)
    chrest_data_path_base = chrest_data_path_base / (str(args.file.stem).replace("*", "") + ".chrest")

    # Save the result data
    chrest_data.save(chrest_data_path_base)
