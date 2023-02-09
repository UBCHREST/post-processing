import argparse
import pathlib
import xml.etree.ElementTree as ET
import h5py

from supportPaths import expand_path


class XdmfGenerator:
    def __init__(self):
        # The xdmf root
        self.xdmf = ET.Element('Xdmf')

        # the grid root, holds the times series
        self.gridCollection = None

        # The time history
        self.timeHistory = []

        # setup the root datastructure
        domain = ET.SubElement(self.xdmf, 'Domain')
        self.gridCollection = ET.SubElement(domain, 'Grid')

        # set this grid to be a time seris
        self.gridCollection.set('CollectionType', 'Temporal')
        self.gridCollection.set('GridType', 'Collection')
        self.gridCollection.set('Name', 'TimeSeries')

    def append_chrest_hdf5(self, hdf5_file):
        # make sure it is a path file
        if isinstance(hdf5_file, str):
            hdf5_file = pathlib.Path(hdf5_file)

        # Load in the hdf5 file
        hdf5 = h5py.File(hdf5_file, 'r')

        # Check each of the cell fields
        hdf5_data = hdf5['data']

        # store the dimensions, note this will be z, y, x
        gridDim = []

        # get the fields
        hdf5_fields = hdf5_data['fields']
        hdf5_grid = hdf5_data['grid']

        # determine the number of dimensions
        dimensions = hdf5_grid['start'].shape[0]

        # create a list of items to output
        fieldNames = []
        for field_name in hdf5_fields.keys():
            # if field data
            if hdf5_fields[field_name].ndim > 1:
                fieldNames.append(field_name)
                # check the dim
                if gridDim:
                    if gridDim != hdf5_fields[field_name].shape[0:dimensions]:
                        raise Exception("All fields must be of same size")
                else:
                    gridDim = hdf5_fields[field_name].shape[0:dimensions]

        # convert the gridDim to a string rep (note add one for cell/face info)
        grid_cell_dim_string = ' '.join(map(lambda d: str(d + 1), gridDim))

        # determine the dim of the grid
        grid_dim = len(gridDim)

        # create a new sub grid
        grid = ET.SubElement(self.gridCollection, 'Grid')
        grid.set('Name', 'mesh')
        grid.set('GridType', 'Uniform')

        # Set the baseline topology
        topology = ET.SubElement(grid, 'Topology')
        topology.set('TopologyType', topology_type[grid_dim])
        topology.set('NumberOfElements', grid_cell_dim_string)

        # add the simple geometry describing a structured grid
        geometry = ET.SubElement(grid, 'Geometry')
        geometry.set('GeometryType', geometry_type[grid_dim])
        write_data_item(geometry, hdf5_grid['start'], hdf5_file.name)
        write_data_item(geometry, hdf5_grid['discretization'], hdf5_file.name)

        # Add in the field data
        for field_name in fieldNames:
            # if there are component names write each component out separately
            hdf5_field_data = hdf5_fields[field_name]
            component_names = None
            if 'components' in hdf5_field_data.attrs:
                component_names = hdf5_field_data.attrs['components'].tolist()

            # write each component as a separate index
            if component_names is not None and len(hdf5_field_data.shape) > 3:
                for component_index in range(len(component_names)):
                    fieldData = ET.SubElement(grid, 'Attribute')
                    fieldData.set('Name', field_name + "_" + component_names[component_index])
                    fieldData.set('Type', 'Scalar')
                    fieldData.set('Center', 'Cell')
                    write_data_hyper_slab_item(fieldData, hdf5_field_data, hdf5_file.name, component_index)
            else:
                fieldData = ET.SubElement(grid, 'Attribute')
                fieldData.set('Name', field_name)
                fieldData.set('AttributeType', 'Scalar')
                fieldData.set('Center', 'Cell')
                write_data_item(fieldData, hdf5_field_data, hdf5_file.name)

        # Store the time
        if hdf5_data.attrs['time'].dtype == float:
            self.timeHistory.append(hdf5_data.attrs['time'])
        else:
            self.timeHistory.append(hdf5_data.attrs['time'][0])

    def write_to_file(self, xdmf_file):
        # add the full time history
        timeHistory = ET.SubElement(self.gridCollection, 'Time')
        timeHistory.set('TimeType', 'List')

        timeArray = ET.SubElement(timeHistory, 'DataItem')
        timeArray.set('Dimensions', str(len(self.timeHistory)))
        timeArray.set('Format', 'XML')
        timeArray.set('NumberType', 'Float')
        timeArray.text = ' '.join(str(e) for e in self.timeHistory)

        # write to file
        tree = ET.ElementTree(self.xdmf)
        try:
            ET.indent(tree, space="\t", level=0)
        except (Exception,):
            print("Could not pretty print xml document, continuing with default format.")

        with open(xdmf_file, 'w') as f:
            f.write('<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            tree.write(f, xml_declaration=False, encoding='unicode')


def write_data_item(xdmf_parent, hdf5_item, hdf5_filename):
    dataItem = ET.SubElement(xdmf_parent, 'DataItem')
    dataItem.set('Dimensions', ' '.join(map(str, hdf5_item.shape)))
    dataItem.set('NumberType', 'Float')
    dataItem.set('Precision', '8')
    dataItem.set('Format', 'HDF')
    dataItem.text = hdf5_filename + ":" + hdf5_item.name


def write_data_hyper_slab_item(xdmf_parent, hdf5_item, hdf5_filename, offset):
    hyperSlabDataItem = ET.SubElement(xdmf_parent, 'DataItem')

    # create an effective size
    effective_size = list(hdf5_item.shape)
    effective_size[-1] = 1

    hyperSlabDataItem.set('Dimensions', ' '.join(map(str, effective_size)))
    hyperSlabDataItem.set('ItemType', 'HyperSlab')
    hyperSlabDataItem.set('Type', 'HyperSlab')

    # Specify the mapping
    coord = []
    # start
    for s in range(len(hdf5_item.shape) - 1):
        coord.append(0)
    coord.append(offset)

    # specify the stride
    for s in range(len(hdf5_item.shape) - 1):
        coord.append(1)
    coord.append(hdf5_item.shape[-1])

    # include the end
    for s in range(len(hdf5_item.shape) - 1):
        coord.append(hdf5_item.shape[s])
    coord.append(1)

    strideDataItem = ET.SubElement(hyperSlabDataItem, 'DataItem')
    strideDataItem.set('Dimensions', f'{3} {len(hdf5_item.shape)}')
    strideDataItem.set('Format', 'XML')
    strideDataItem.text = ' '.join(map(str, coord))

    # add the original data item
    write_data_item(hyperSlabDataItem, hdf5_item, hdf5_filename)


# store some dimensional information
topology_type = {
    2: "2DCoRectMesh",
    3: "3DCoRectMesh"
}

geometry_type = {
    2: "Origin_DxDy",
    3: "Origin_DxDyDz"
}

# Main function to parser input files and run the document generator
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an xdmf file from MatLab data files holding structed data. '
                                                 'See https://github.com/cet-lab/experimental-post-processing/wiki'
                                                 '/Matlab-To-XdmfGenerator for details.  ')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The path to the hdf5 file(s) containing the structured data.  A wild card can be used '
                             'to supply more than one file.')
    args = parser.parse_args()

    # Get the parent file
    xdmf_file = args.hdf5_file.parent / (args.hdf5_file.stem.replace('*', '') + ".xdmf")

    # expand any wild cards in the path name
    hdf5_paths = expand_path(args.hdf5_file)

    # generate an xdfm object
    xdfm = XdmfGenerator()

    # convert with path
    for hdf5_file in hdf5_paths:
        # create component markdown
        xdfm.append_chrest_hdf5(hdf5_file)

    # write the xdmf file
    xdfm.write_to_file(xdmf_file)
