import argparse
import pathlib
import sys, os
import csv

import numpy as np
import h5py

from chrestData import ChrestData
from supportPaths import expand_path
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import interpolate 
from enum import Enum
from xdmfGenerator import XdmfGenerator

# class syntax
class FieldType(Enum):
    CELL = 1
    VERTEX = 2


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
        self.index_per_time = dict()
        self.timeitervals = []
        self.numintervals = 1
        self.times = []

        # store the cells and vertices
        self.cells = None
        self.vertices = None
        self.geometry_file = 0
        
        # stroe interpolation information
        self.vtx=[]
        self.wts=[]
        self.Wx=[]
        self.Wy=[]
        self.Wz=[]

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
                self.geometry_file = file

            # extract the time
            timeInFile = hdf5['time'][:, 0]
            for idx, time in enumerate(timeInFile):
                self.times.append(time)
                self.files_per_time[time] = file
                self.index_per_time[time] = idx

        # Store the list of times
        self.times = list(self.files_per_time.keys())
        self.times.sort()
        self.timeitervals.append(self.times.copy())
        
        self.interpolate=False
        self.surface=False
        self.gradients=[]
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
    def get_fields(self, field_type=FieldType.CELL):
        fields_names = []

        if len(self.files) > 0:
            with h5py.File(self.files[0], 'r') as hdf5:
                if 'cell_fields' in hdf5 and field_type == FieldType.CELL:
                    fields_names.extend(hdf5['cell_fields'].keys())
                if 'vertex_fields' in hdf5 and field_type == FieldType.VERTEX:
                    fields_names.extend(hdf5['vertex_fields'].keys())
        return fields_names

    """
    computes the cell center for each cell [c, d]
    """

    def compute_cell_centers(self, dimensions=-1):
        # create a new np array based upon the dim
        number_cells = self.cells.shape[0]
        vertices = self.vertices[:]
        vertices_dim = 1
        if len(vertices.shape) > 1:
            vertices_dim = vertices.shape[1]
        if dimensions < 0:
            dimensions = vertices_dim

        coords = np.zeros((number_cells, dimensions))

        # march over each cell
        for c in range(len(coords)):
            cell_vertices = vertices.take(self.cells[c], axis=0)

            # take the average
            cell_center = np.sum(cell_vertices, axis=0)
            cell_center = cell_center / len(cell_vertices)

            # put back
            coords[c, 0:vertices_dim] = cell_center

        # if this is one d, flatten
        if dimensions == 1:
            coords = coords[:, 0]

        return coords
    """
    Returns the vertexes in the order used in the mesh
    """

    def compute_vertexes(self):
        return self.vertices

    """
    Returns either the cell centers or vertices 
    """
    def compute_geometry(self, field_type, geometry_time=0.0):
        # if a specific time is specified, reload the geometry from that file
        if self.files_per_time[geometry_time] != self.geometry_file:
            # Load in the hdf5 file
            hdf5 = h5py.File(self.files_per_time[geometry_time], 'r')

            self.cells = hdf5["viz"]["topology"]["cells"]
            self.vertices = hdf5["geometry"]["vertices"][:]
            self.geometry_file = self.files_per_time[geometry_time]

        if field_type == FieldType.CELL:
            return self.compute_cell_centers()
        elif field_type == FieldType.VERTEX:
            return self.compute_vertexes()
        else:
            return None

    """
    gets the specified field and the number of components
    """

    def get_field(self, field_name, interwal=0, component_names=None):
        # create a dictionary of times/data
        data = []
        components = 0
        all_component_names = None
        # march over the files
        for t in self.timeitervals[interwal]:
            # for i in range(start,stop):
            # Load in the file
            with h5py.File(self.files_per_time[t], 'r') as hdf5:
                try:
                    # Check each of the cell fields
                    if 'cell_fields' in hdf5 and field_name in hdf5['cell_fields']:
                        hdf5_field = hdf5['cell_fields'][field_name]
                    elif 'vertex_fields' in hdf5 and field_name in hdf5['vertex_fields']:
                        hdf5_field = hdf5['vertex_fields'][field_name]

                    # if there are multiple times in this file, extract them
                    time_index = self.index_per_time[t]

                    if len(hdf5_field[time_index, :].shape) > 1:
                        components = hdf5_field[time_index, :].shape[1]
                        # check for component names
                        all_component_names = [""] * components

                        for i in range(components):
                            if ('componentName' + str(i)) in hdf5_field.attrs:
                                all_component_names[i] = hdf5_field.attrs['componentName' + str(i)].decode("utf-8")

                    # load in the specified field name, but check if we should down select components
                    if component_names is None:
                        hdf_field_data = hdf5_field[time_index, :]
                        data.append(hdf_field_data)
                    else:
                        # get the indexes
                        indexes = []
                        for component_name in component_names:
                            try:
                                indexes.append(all_component_names.index(component_name))
                            except ValueError as error:
                                raise Exception("Cannot locate " + component_name + " in field " + field_name + ". ",
                                                error)

                        # sort the index list in order
                        index_name_list = zip(indexes, component_names)
                        index_name_list = sorted(index_name_list)
                        indexes, component_names = zip(*index_name_list)

                        # down select only the components that were asked for
                        hdf_field_data = hdf5_field[time_index, ..., list(indexes)]
                        data.append(hdf_field_data)
                        all_component_names = list(component_names)
                        components = len(all_component_names)
                except Exception as e:
                    raise Exception(
                        "Unable to open field " + field_name + "." + str(e))
            hdf5.close

        return np.stack(data), components, all_component_names

    def sort_time(self, inputn, filerange):
        if int(filerange[1]) != -1:
            vector = self.times[int(filerange[0]):int(filerange[1]) + 1]
        elif int(filerange[1]) != 0:
            vector = self.times[int(filerange[0] - 1):int(filerange[1])]
        else:
            vector = self.times
        n = int(inputn)
        self.numintervals = (len(vector) // n)   # Calculate the number of sub-vectors
    
        self.timeitervals = [vector[i:i + n] for i in range(0, self.numintervals * n, n)]
    
        # If there are any remaining elements, create a sub-vector with the leftover elements
        remaining_elements = len(vector) % n
        if remaining_elements != 0:
            self.timeitervals.append(vector[self.numintervals * n:])
            self.numintervals += 1
        # if remaining_elements == 0:
        #     self.numintervals += 1
        # return sub_vectors
    

    """
    converts the supplied fields ablate object to chrest data object.
    """
        
    def map_to_chrest_data(self, chrest_data, field_mapping, iteration, field_select_components=dict(),
                            max_distance=sys.float_info.max):
        # get the cell centers for this mesh
        # cell_centers = self.compute_cell_centers(chrest_data.dimensions)
        gradientfield=[]
        filename_coordinate = 'ablatecoords_'+str(len(self.vertices))+'_vert.csv'
        # filename2 = 'verticies_ablate_'+str(len(xyz))+'chrest_'+str(len(uvw))+'.csv'
        filepath = self.files[0].parent / filename_coordinate
        
        if not filepath.exists():        
            cell_centers = self.compute_cell_centers(chrest_data.dimensions)
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(cell_centers)
            print(f'Saved {filename_coordinate} successfully.')
        else:
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                cell_centers = np.array([[float(value) for value in row] for row in reader])
            print(f'Read {filename_coordinate} successfully.')
            
        # add the ablate times to chrest
        # chrest_data.times = self.times.copy()
        chrest_data.times = self.timeitervals[iteration].copy()

        # store the fields in ablate we need
        ablate_fields = list(field_mapping.keys())

        # store the chrest data in the same order
        ablate_field_data = []
        chrest_field_data = []
        components = []

        # create the new field in the chrest data
        
        for ablate_field in ablate_fields:
            chrest_field = field_mapping[ablate_field]

            # check to see if there is a down selection of components
            component_select_names = field_select_components.get(ablate_field)

            # get the field from ablate
            ablate_field_data_tmp, components_tmp, component_names = self.get_field(ablate_field, iteration,
                                                                                    component_select_names)
            ablate_field_data.append(ablate_field_data_tmp)
            components.append(components_tmp)
            chrest_field_data.append(chrest_data.create_field(chrest_field, components_tmp, component_names)[0])
            gradientfield.append(0)
        
        #Add aditional derivative fields
        # derivative_component=['aux_temperature','aux_velocity']
        # for derfield in derivative_component:
        for derfield in self.gradients:    
            chrest_field = field_mapping[derfield]
            chrest_field ='d'+chrest_field 
            # check to see if there is a down selection of components
            component_select_names = field_select_components.get(derfield)

            # get the field from ablate
            ablate_field_data_tmp, components_tmp, component_names = self.get_field(derfield,iteration,
                                                                                    component_select_names)
            if component_names is None:
                ablate_field_data.append(ablate_field_data_tmp)
                components.append(3)
                component_names=[]
                component_names.append(chrest_field+ '_dx')
                component_names.append(chrest_field+ '_dy')
                component_names.append(chrest_field+ '_dz')
                chrest_field_data.append(chrest_data.create_field(chrest_field, 3, component_names)[0])
                gradientfield.append(1)
            else:
                idx=0

                for compfield in component_names:
                    ablate_field_data.append(ablate_field_data_tmp[:,:,idx])
                    components.append(3)
                    component_names=[]
                    comfieldname='d'+compfield
                    component_names.append(comfieldname+ '_dx')
                    component_names.append(comfieldname+ '_dy')
                    component_names.append(comfieldname+ '_dz')
                    chrest_field_data.append(chrest_data.create_field(comfieldname, 3, component_names)[0])
                    idx+=1
                    gradientfield.append(1)
        
        # build a list of k, j, i points to iterate over
        chrest_coords = chrest_data.get_coordinates()

        # size up the storage value
        chrest_cell_number = np.prod(chrest_data.grid)

        # reshape to get a single list order back
        chrest_cell_centers = chrest_coords.reshape((chrest_cell_number, chrest_data.dimensions))

        # now search and copy over data
        tree = KDTree(cell_centers)
        dist, points = tree.query(chrest_cell_centers, workers=-1)
        
        if self.surface:
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
        
        else:
            if self.interpolate:
                if (len(self.vtx) == 0): 
                    if self.gradients is None:
                        self.vtx, self.wts = interpolate.calc_interpweights_3D(cell_centers, chrest_cell_centers,self.files[0].parent)
                    elif self.gradients is not None:
                        self.vtx, self.wts, self.Wx, self.Wy, self.Wz = interpolate.calc_allweights_3D(cell_centers, chrest_cell_centers,self.files[0].parent)
            
            if np.size(cell_centers,1)!=3:
                print('Warning! The code has not been tested for 2D grids...')
            
            # march over each field
            idx = 0
            for f in range(len(ablate_field_data)):
                if not gradientfield[f]:
                    if self.interpolate:
                        #interpolation (calculate or use precomputed weights)
                        #ps. only do interpolation on points that are ~inside of the domain
                        #currently ONLY 3D interpolation is supported
                        ablate_field_in_chrest_order = np.zeros_like(ablate_field_data[f][:, points])
                        for t in range(len(ablate_field_in_chrest_order)):
                            if len(ablate_field_data[f].shape)>2:
                                for ii in range(ablate_field_data[f].shape[2]):
                                    fielddata=ablate_field_data[f][t]
                                    ablate_field_in_chrest_order[t,:,ii] = np.reshape(interpolate.interpolate3D(self.vtx,self.wts,fielddata[:,ii],points),(1,self.vtx.shape[0]))
                            else:
                                fielddata=np.transpose(ablate_field_data[f][:])
                                ablate_field_in_chrest_order[t,:] = np.reshape(interpolate.interpolate3D(self.vtx,self.wts,fielddata[:,t],points),(1,self.vtx.shape[0]))
                    else:  
                        # closest node 
                        ablate_field_in_chrest_order = ablate_field_data[f][:, points]
                    
                    setToZero = np.where(dist > max_distance)
                    closestpoints=(np.array(points[setToZero]),)
                    ablate_field_in_chrest_order[:, setToZero] = ablate_field_data[f][:, closestpoints]
                    
                    # reshape it back to k,j,i
                    ablate_field_in_chrest_order = ablate_field_in_chrest_order.reshape(
                        chrest_field_data[f].shape
                    )
                    
                    # copy over the data
                    chrest_field_data[f][:] = ablate_field_in_chrest_order[:]
                
                else:
                    # calculating the gradients
                    ablate_field_in_chrest_order = np.zeros((ablate_field_data[f][:, points].shape[0],ablate_field_data[f][:, points].shape[1],3))
                    for t in range(len(ablate_field_in_chrest_order)):
                        fielddata=ablate_field_data[f][t]
                        
                        ablate_field_in_chrest_order[t,:,0] = np.reshape(interpolate.grad(fielddata[:],self.vtx,self.Wx),(1,self.vtx.shape[0]))
                        ablate_field_in_chrest_order[t,:,1] = np.reshape(interpolate.grad(fielddata[:],self.vtx,self.Wy),(1,self.vtx.shape[0]))
                        ablate_field_in_chrest_order[t,:,2] = np.reshape(interpolate.grad(fielddata[:],self.vtx,self.Wz),(1,self.vtx.shape[0]))
                    
                    setToZero = np.where(dist > max_distance)
                    ablate_field_in_chrest_order[:, setToZero] = 0.0               
                    
                    # using scipy built in interpolator, impractical for normal size grids, takes really long...
                    # ablate_field_in_chrest_order = griddata((cell_centers[:,0],cell_centers[:,1],cell_centers[:,2]), ablate_field_data[f].T, (chrest_cell_centers), method='linear')
        
                    # reshape it back to k,j,i
                    ablate_field_in_chrest_order = ablate_field_in_chrest_order.reshape(
                        chrest_field_data[f].shape
                    )
                    
                    # copy over the data
                    chrest_field_data[f][:] = ablate_field_in_chrest_order[:]
                
                idx+=1
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
                        nargs='+', default=[0., 0.0, -0.0128]
                        )

    parser.add_argument('--end', dest='end', type=float,
                        help='Optional ending point for chrest data. Default is [0.1, 0.0254, 0.0127]',
                        nargs='+', default=[0.1, 0.0256, 0.0128]
                        )

    parser.add_argument('--delta', dest='delta', type=float,
                        help='Optional grid spacing for chrest data. Default is [0.0005, 0.0005, 0.0005]',
                        nargs='+', default=[0.00025, 0.00025, 0.00025]
                        )

    parser.add_argument('--fields', dest='fields', type=str,
                        help='The list of fields to map from ablate to chrest in format  --field '
                             'ablate_name:chrest_name:components --field aux_temperature:temperature '
                             'aux_velocity:vel. Components are optional list such as H2,N2',
                        nargs='+', default=["aux_temperature:temperature", "aux_velocity:vel"]
                        )

    parser.add_argument('--print_fields', dest='print_fields', action='store_true',
                        help='If true, prints the fields available to convert', default=False
                        )

    parser.add_argument('--max_distance', dest='max_distance', type=float,
                        help="The max distance to search for a point in ablate", default=sys.float_info.max)

    parser.add_argument('--batchsize', dest='batchsize', type=float,
                        help="The number of files to be loaded in at once")

    parser.add_argument('--filerange', dest='filerange', type=float,
                        help="The first and last files that the user want to process in the directory default is: [0 -1]",
                        nargs='+')
    
    parser.add_argument('--interpolate', dest='interpolate', action='store_true',
                        help="Whether you would like to interpolate or not. Currently only available for 3D.",
                        )
    parser.add_argument('--surface', dest='surf', action='store_true',
                        help="Whether you are trying to access files from the surface monitor.",
                        )    
    
    parser.add_argument('--gradients', dest='gradients', type=str,
                        help='The list of fields in ablate output format  --gradients '
                             'eg: aux_temperature aux_velocity . The code takes all 3 (x,y,z) spatial derivatives',
                        nargs='+'
                        )    

    args = parser.parse_args()

    # this is some example code for chest file post-processing
    ablate_data = AblateData(args.file)
    print('Starting the conversion to chrest format.')
    if args.print_fields:
        print("Available fields: ", ', '.join(ablate_data.get_fields()))

    # list the fields to map
    field_mappings = dict()
    component_select_names = dict()
    for field_mapping in args.fields:
        field_mapping_list = field_mapping.split(':')
        field_mappings[field_mapping_list[0]] = field_mapping_list[1]

        # check to see if there are select components
        if len(field_mapping_list) > 2:
            component_select_names[field_mapping_list[0]] = field_mapping_list[2].split(',')
            
    #determine if you want ot interpolate
    ablate_data.interpolate=int(args.interpolate)

    #determine if you want ot interpolate
    ablate_data.surface=int(args.surf)    
    
    #determine fields for gradients
    if args.gradients is not None:
        ablate_data.gradients=args.gradients      
        
    # create a chrest data
    chrest_data = ChrestData()
    chrest_data.setup_new_grid(args.start, args.end, args.delta)
    
    if args.filerange is None:
        filerange=[0,len(ablate_data.times)]
        startind=0
    else:
        filerange=args.filerange
        startind=filerange[0]
        
    if args.batchsize is not None:
        if len(ablate_data.times) > args.batchsize:
            ablate_data.sort_time(args.batchsize,filerange)
            print("The code processes " + str(args.batchsize) + " files at a time.")
        else:
            ablate_data.sort_time(len(ablate_data.times),filerange)
            print("The code processes " + str(len(ablate_data.times)) + " files at a time.")
    else:
        ablate_data.sort_time(len(ablate_data.times),filerange)
        
    import time
    for i in range(0,ablate_data.numintervals):
        # map the ablate data to chrest
        
        start_time = time.time()
        ablate_data.map_to_chrest_data(chrest_data, field_mappings,i, component_select_names, args.max_distance)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        # write the new file without wild card
        chrest_data_path_base = args.file.parent / (str(args.file.stem).replace("*", "") + ".chrest")
        chrest_data_path_base.mkdir(parents=True, exist_ok=True)
        chrest_data_path_base = chrest_data_path_base / (str(args.file.stem).replace("*", "") + ".chrest")

        # Save the result data
        chrest_data.savepart(chrest_data_path_base, i, len(ablate_data.timeitervals[0]), startind)
    
    newdir = args.file.parent / (str(args.file.stem).replace("*", "") + ".chrest")
    xdmf_file =  newdir / (str(args.file.stem.replace('*', '') + ".xdmf"))

    hdf5_paths = expand_path(newdir / os.path.basename(str(args.file)))

    # generate an xdfm object
    xdfm = XdmfGenerator()

    # #convert with new path
    for hdf5_file in hdf5_paths:
        # create component markdown
        xdfm.append_chrest_hdf5(hdf5_file)

    # write the xdmf file
    xdfm.write_to_file(xdmf_file)
    
