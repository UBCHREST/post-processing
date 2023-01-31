import argparse
import pathlib

import h5py

from support.supportPaths import expand_path


class ChrestData:
    # a list of chrest files
    files = None

    # store the files based upon time
    filesPerTime = dict()

    # store the list of fields available
    fields = []

    # load the metadata from the first file
    metadata = dict()

    def __init__(self, files):
        if isinstance(files, str):
            self.files = expand_path(files)
        elif isinstance(files, pathlib.Path):
            self.files = expand_path(files)
        else:
            self.files = files

        #Open each file to get the time and check the available fields
        for file in self.files:
            # Load in the hdf5 file
            hdf5 = h5py.File(file, 'r')


            # Check each of the cell fields
            hdf5_data = hdf5['data']
            hdf5_fields = hdf5_data['fields']

            if len(self.fields):
                # check for uniform values
                if self.fields != list(hdf5_fields.keys()):
                    raise Exception("There is an error in the chrest files.  The fields in each file do not match.")
            else:
                # If it is empty just end items
                self.fields = list(hdf5_fields.keys())

                # get the fields and other data
            time = hdf5_data.attrs['time'][0]
            self.filesPerTime[time] = file

            # Load in the metadata
            if ~ len(self.metadata):
                for (key, value) in hdf5_data.attrs.items():
                    if h5py.check_string_dtype(value.dtype):
                        self.metadata[key] = value[0]

# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an xdmf file from MatLab data files holding structed data. '
                                                 'See https://github.com/cet-lab/experimental-post-processing/wiki'
                                                 '/Matlab-To-XdmfGenerator for details.  ')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The path to the hdf5 file(s) containing the structured data.  A wild card can be used '
                             'to supply more than one file.')
    args = parser.parse_args()

    # this is some example code for chest file post processing
    chrestData = ChrestData(args.hdf5_file)
