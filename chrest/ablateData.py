import argparse
import pathlib
import sys

import numpy as np
import h5py

from chrestData import ChrestData
from supportPaths import expand_path
from scipy.spatial import KDTree


class AblateData:
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

        # store the files based upon time
        self.files_per_time = dict()
        self.times = []

        # store the cells and vertices
        self.cells = None
        self.vertices = None

        # load the metadata from the first file
        self.metadata = dict()

        # Open each file to get the time and check the available fields
        for file in self.files:
            # Load in the hdf5 file
            hdf5 = h5py.File(file, 'r')

            # If not set, copy over the cells and vertices
            if self.cells is None:
                self.cells = hdf5["viz"]["topology"]["cells"]
                self.vertices = hdf5["geometry"]["vertices"][:]

            # extract the time
            time = hdf5['time'][0][0]
            self.times.append(time)

            self.files_per_time[time] = file

        # Store the list of times
        self.times = list(self.files_per_time.keys())
        self.times.sort()

        # Load in any available metadata based upon the file structure
        self.metadata['path'] = str(self.files[0])

        # load in the restart file
        try:
            restart_file = self.files[0].parent.parent / "restart.rst"
            if restart_file.is_file():
                self.metadata['restart.rst'] = restart_file.read_text()
        except (Exception,):
            print("Could not locate restart.rst for metadata")

        # load in the restart file
        try:
            simulation_directory = self.files[0].parent.parent
            yaml_files = list(simulation_directory.glob("*.yaml"))
            for yaml_file in yaml_files:
                self.metadata[yaml_file.name] = yaml_file.read_text()
        except (Exception,):
            print("Could not locate yaml files for metadata")

    """
    Reports the fields from each the file
    """

    def get_fields(self):
        fields_names = []

        if len(self.files) > 0:
            with h5py.File(self.files[0], 'r') as hdf5:
                fields_names.extend(hdf5['cell_fields'].keys())

        return fields_names


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

    def map_to_chrest_data(self, chrest_data, field_mapping, max_distance=sys.float_info.max):
        # get the cell centers for this mesh
        cell_centers = self.compute_cell_centers(chrest_data.dimensions)

        # add the ablate times to chrest
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
            ablate_field_data_tmp, components_tmp = self.get_field(ablate_field)

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
        dist, points = tree.query(chrest_cell_centers)

        # march over each field
        for f in range(len(ablate_field_data)):
            # get in the correct order
            ablate_field_in_chrest_order = ablate_field_data[f][:, points]

            setToZero = np.where(dist > max_distance)

            ablate_field_in_chrest_order[:, setToZero] = 0.0

            # reshape it back to k,j,i
            ablate_field_in_chrest_order = ablate_field_in_chrest_order.reshape(
                chrest_field_data[f].shape
            )

            # copy over the data
            chrest_field_data[f][:] = ablate_field_in_chrest_order[:]

        # copy over the metadata
        chrest_data.metadata = self.metadata


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a chrest data file from an ablate file')
    parser.add_argument('--file', dest='file', type=pathlib.Path, required=True,
                        help='The path to the ablate hdf5 file(s) containing the ablate data.'
                             '  A wild card can be used '
                             'to supply more than one file.')
    parser.add_argument('--start', dest='start', type=float,
                        help='Optional starting point for chrest data. Default is [0., 0.0, -0.0127]',
                        nargs='+', default=[0., 0.0, -0.0127]
                        )

    parser.add_argument('--end', dest='end', type=float,
                        help='Optional ending point for chrest data. Default is [0.1, 0.0254, 0.0127]',
                        nargs='+', default=[0.1, 0.0254, 0.0127]
                        )

    parser.add_argument('--delta', dest='delta', type=float,
                        help='Optional grid spacing for chrest data. Default is [0.0005, 0.0005, 0.0005]',
                        nargs='+', default=[0.0005, 0.0005, 0.0005]
                        )

    parser.add_argument('--fields', dest='fields', type=str,
                        help='The list of fields to map from ablate to chrest in format --field aux_temperature:temperature aux_velocity:vel',
                        nargs='+', default=["aux_temperature:temperature", "aux_velocity:vel"]
                        )

    parser.add_argument('--remove_coord', dest='remove_coord', type=int,
                        help="Remove a coordinate direction (i.e. x,y,z to x,z)",
                        nargs='+', default=[])

    parser.add_argument('--max_distance', dest='max_distance', type=float,
                        help="The max distance to search for a point in ablate", default=sys.float_info.max)

    args = parser.parse_args()

    # this is some example code for chest file post-processing
    ablate_data = AblateData(args.file)

    if args.print_fields:
        print("Available fields: ", ', '.join(ablate_data.get_fields()))

    # list the fields to map
    field_mappings = dict()
    for field_mapping in args.fields:
        field_mapping_list = field_mapping.split(':')
        field_mappings[field_mapping_list[0]] = field_mapping_list[1]

    # create a chrest data
    chrest_data = ChrestData()
    chrest_data.setup_new_grid(args.start, args.end, args.delta)

    # map the ablate data to chrest
    ablate_data.map_to_chrest_data(chrest_data, field_mappings, args.max_distance)

    # write the new file without wild card
    chrest_data_path_base = args.file.parent / (str(args.file.stem).replace("*", "") + ".chrest")
    chrest_data_path_base.mkdir(parents=True, exist_ok=True)
    chrest_data_path_base = chrest_data_path_base / (str(args.file.stem).replace("*", "") + ".chrest")

    # Save the result data
    chrest_data.save(chrest_data_path_base)
