# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:50:30 2023

@author: rkolo
"""

#vtcp

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

import matplotlib.pyplot as plt
import matplotlib as mpl

# class syntax
class FieldType(Enum):
    CELL = 1
    VERTEX = 2


class VTCP:
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
        
        #Store some soot constants
        n=2.31
        k=0.71
        self.C2=1.4388e-2
        self.C0=36*np.pi*n*k/((n**2-k**2+2)**2 + 4*n**2*k**2)
        self.C0_red=4.24
        self.C0_green=4.55
        self.lambda_red=650e-9
        self.lambda_green=532e-9
        self.lambda_blue=410e-9
        self.SBC=5.6696e-8 #Stephan-Boltzman constant
        self.c = 3e8
        self.h = 6.626e-34
        
        
        self.cratio=np.log(4.24/4.55)
        self.lambdaratio=np.log((self.lambda_green/self.lambda_red)**6)
        
        self.I=[]
        
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
        
    def convertfield(self, chrest_data, field_mapping, iteration, field_select_components=dict(),
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
        
        #change this to required fields
        # ablate_fields=['aux_temperature','solution_densityYi:densityYi:C(S)']
        
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
        

        
        # build a list of k, j, i points to iterate over
        chrest_coords = chrest_data.get_coordinates()

        # size up the storage value
        chrest_cell_number = np.prod(chrest_data.grid)

        # reshape to get a single list order back
        chrest_cell_centers = chrest_coords.reshape((chrest_cell_number, chrest_data.dimensions))

        # now search and copy over data
        tree = KDTree(cell_centers)
        dist, points = tree.query(chrest_cell_centers, workers=-1)
        
        if len(self.vtx) == 0: 
            if self.interpolate and self.gradients is None:
                self.vtx, self.wts = interpolate.calc_interpweights_3D(cell_centers, chrest_cell_centers,self.files[0].parent)
            elif self.gradients is not None:
                self.vtx, self.wts, self.Wx, self.Wy, self.Wz = interpolate.calc_allweights_3D(cell_centers, chrest_cell_centers,self.files[0].parent)
        
        if np.size(cell_centers,1)!=3:
            print('Warning! The code has not been tested for 2D grids...')
        
        # march over each field
        for f in range(len(ablate_field_data)):
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


        # copy over the metadata
        chrest_data.metadata = self.metadata
        self.raw_data=chrest_data
    
    def trace_rays(self,chrest_data):
        
        Temp=np.copy(chrest_data.new_data['temperature'])
        rhoYc=np.copy(chrest_data.new_data['densityYi'])
        fv=rhoYc/2000
        indx=3
        indy=2
        indz=1
        
        # for debug purposes
        # Temp[:]= 2000
        fv[:] = 1e-6
        
        
        I=np.zeros([np.shape(Temp)[0],np.shape(Temp)[indx],np.shape(Temp)[indy],4])        
        
        
        #this code assumes everything is in chrest fromat thus dx=0.25mm
        dx=chrest_data.delta[0]
        self.dx=dx
        
        for t in range(np.shape(Temp)[0]):
            for i in range(np.shape(Temp)[indx]):
                for j in range(np.shape(Temp)[indy]):
                    Itrace=0
                    Itrace_red=0
                    Itrace_green=0
                    Itrace_blue=0
                    Kappa=[0,0,0,0]
                    blackbody=[0,0,0,0]
                    expofunc=[0,0,0,0]
                    
                    for k in range(np.shape(Temp)[indz]):
                        # if Temp[t,k,j,i]==0:
                        #     print('temperature is 0 at i='+ str(i)+' j='+str(j)+' k='+str(k))
                        Kappa[0]=3.72*fv[t,k,j,i]*self.C0*Temp[t,k,j,i]/self.C2
                        Kappa[1]=self.C0*fv[t,k,j,i]/self.lambda_red
                        Kappa[2]=self.C0*fv[t,k,j,i]/self.lambda_green
                        Kappa[3]=self.C0*fv[t,k,j,i]/self.lambda_blue
                        
                        # dx=0.025
                        
                        expofunc[0] = np.exp(-1*(Kappa[0]*dx))
                        expofunc[1] = np.exp(-1*(Kappa[1]*dx))
                        expofunc[2] = np.exp(-1*(Kappa[2]*dx))
                        expofunc[3] = np.exp(-1*(Kappa[3]*dx))
                        
                        blackbody[0] = self.SBC*Temp[t,k,j,i]**4 / np.pi
                        blackbody[1] = (2*self.c**2)/(self.lambda_red**5*np.exp(self.C2/(self.lambda_red*Temp[t,k,j,i]))) 
                        blackbody[2] = (2*self.c**2)/(self.lambda_green**5*np.exp(self.C2/(self.lambda_green*Temp[t,k,j,i])))
                        blackbody[3] = (2*self.c**2)/(self.lambda_blue**5*np.exp(self.C2/(self.lambda_blue*Temp[t,k,j,i])))
                        
                        
                        # Itrace.append(self.SBC*Temp[t,i,j,k]**4/np.pi*(1-np.exp(-Kappa*dx)))
                        Itrace = Itrace*expofunc[0] + blackbody[0]*(1-expofunc[0])
                        Itrace_red = Itrace_red*expofunc[1] + blackbody[1]*(1-expofunc[1])
                        Itrace_green = Itrace_green*expofunc[2] + blackbody[2]*(1-expofunc[2])
                        Itrace_blue = Itrace_blue*expofunc[3] + blackbody[3]*(1-expofunc[3])
                        
                    I[t,i,j,:]=[Itrace,Itrace_red,Itrace_green,Itrace_blue]
        
        self.I=I

        
    def get_temperature(self,OutputDirectory,ind,intlen,startind,ShowPlots=False):

        
        tcp_temperature=np.zeros([np.shape(self.I)[1],np.shape(self.I)[2]])
        
        threshold_fraction = 0.05  # Threshold for the absolute intensity

        for t in range(np.shape(self.I)[0]):
            for i in range(np.shape(self.I)[1]):
                for j in range(np.shape(self.I)[2]):
                    
                    tcp_temperature[i,j]=self.C2*(1/self.lambda_red - 1/self.lambda_green)/(np.log(self.I[t,i,j,2]/self.I[t,i,j,1])+self.cratio+self.lambdaratio)
            saveing=t+ind*intlen+int(startind)
            self.plot_temperature(tcp_temperature,OutputDirectory / f'Temperature{t}.png', ShowPlots)
            self.savecsv(tcp_temperature,OutputDirectory / f'Temperature{t}.csv')

                    


            # ratio = data[1, :, :, :, :] / data[0, :, :, :, :]
            # ratio = np.nan_to_num(ratio)
            # tcp_temperature = np.zeros_like(ratio, dtype=np.dtype(float))

            # for n in range(np.shape(data)[1]):
            #     for i in range(np.shape(data)[2]):
            #         for j in range(np.shape(data)[3]):
            #             for k in range(np.shape(data)[4]):
            #                 if data[0, n, i, j, k] < threshold_fraction * np.max(data[0, n, :, :, :]) \
            #                         or data[1, n, i, j, k] < threshold_fraction * np.max(data[1, n, :, :, :]):
            #                     tcp_temperature[n, i, j, k] = 0
            #                 elif ratio[n, i, j, k] != 0:
            #                     tcp_temperature[n, i, j, k] = (c2 * ((1. / lambdaR) - (1. / lambdaG))) / (
            #                             np.log(ratio[n, i, j, k]) + np.log((lambdaG / lambdaR) ** 5))

            #                 if tcp_temperature[n, i, j, k] < 300:
            #                     tcp_temperature[n, i, j, k] = 300

            # # Assign the computed temperatures to the corresponding class variable
            # if data_type == 'front':
            #     self.front_tcp_temperature = tcp_temperature
            # else:
            #     self.top_tcp_temperature = tcp_temperature

    # Get the size of a single mesh.
    # Iterate through the time steps
    # Iterate through each time step and place a point on the plot
    
    # def get_optical_thickness(self, dns_data):
    #     # Calculate the optical thickness of the frame
    #     # First get the absorption for each cell in the dns
    #     dns_temperature, _, _ = dns_data.get_field(self.dns_temperature_name)
    #     if self.dns_soot is None:
    #         self.get_dns_soot(dns_data)
    #     kappa = (3.72 * self.dns_soot * self.C_0 * dns_temperature) / self.C_2  # Soot mean absorption
    #     # Then sum the absorption through all cells in the ray line
    #     axis_values = [1, 2]
    #     optical_thickness_attributes = ['front_dns_optical_thickness', 'top_dns_optical_thickness']

    #     for axis, attribute in zip(axis_values, optical_thickness_attributes):
    #         dns_sum_soot = kappa.sum(axis=axis, keepdims=True)
    #         setattr(self, attribute, dns_sum_soot * dns_data.delta[2 - axis])
            
    
            
    def plot_temperature(self,T,outputFileName,ShowPlot):
        
        xvect=np.arange(0,T.shape[0]*self.dx,self.dx)
        yvect=np.arange(0,T.shape[1]*self.dx,self.dx)
        print('The temperature is '+str(T[0,0]))

        T[T==0]=np.nan
        mpl.rcParams['axes.linewidth'] = 2.5  # set the value globally
        mpl.rcParams.update({'figure.autolayout': True})
        from matplotlib.ticker import AutoMinorLocator
        from matplotlib import cm
        fig, ax = plt.subplots(figsize=(20, 5))
        levels = np.linspace(500,3500,100)
        cp = ax.contourf(xvect, yvect,np.transpose(T),levels,cmap = cm.hot)
        cbar = fig.colorbar(cp)
        cbar.set_label(label='Temperature',size=22)
        cbar.ax.yaxis.set_tick_params(length=10,width=2.5,labelsize=18)
        ax.set_xlabel("$x (mm) $", fontsize=22, fontname="Arial")
        ax.set_ylabel("y (mm)", fontsize=22, fontname="Arial")         
        ax.set_title('Virutal TCP', fontsize=22)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_tick_params(length=10, width=2.5, labelsize=18)
        ax.yaxis.set_tick_params(length=10, width=2.5, labelsize=18)
        ax.xaxis.set_tick_params('minor', length=5, width=2.5)
        ax.yaxis.set_tick_params('minor', length=5, width=2.5)
        ax.grid()
        if(ShowPlot):
            plt.show()
        plt.savefig(outputFileName)
        
    def savecsv(self,T,outputFileName):
        with open(outputFileName, "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(T)

      
    def saveh5files(self):
        
        return 0


# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a chrest data file from an ablate file')
    parser.add_argument('--file', dest='file', type=pathlib.Path, required=True,
                        help='The path to the ablate hdf5 file(s) containing the ablate data.'
                             '  A wild card can be used '
                             'to supply more than one file.')

    parser.add_argument('--max_distance', dest='max_distance', type=float,
                        help="The max distance to search for a point in ablate", default=sys.float_info.max)

    parser.add_argument('--batchsize', dest='batchsize', type=float,
                        help="The number of files to be loaded in at once")

    parser.add_argument('--filerange', dest='filerange', type=float,
                        help="The first and last files that the user want to process in the directory default is: [0 -1]",
                        nargs='+')
   

    args = parser.parse_args()

    # this is some example code for chest file post-processing
    vtcp_data = VTCP(args.file)
    print('Starting virtual tcp.')


    # list the fields to map
    field_mappings = dict()
    component_select_names = dict()
    fields=['aux_temperature:temperature', 'solution_densityYi:densityYi:C(S)']
    for field_mapping in fields:
        field_mapping_list = field_mapping.split(':')
        field_mappings[field_mapping_list[0]] = field_mapping_list[1]

        # check to see if there are select components
        if len(field_mapping_list) > 2:
            component_select_names[field_mapping_list[0]] = field_mapping_list[2].split(',')
     
        
    # create a chrest data
    delta=[0.001, 0.001, 0.001]
    end=[0.1, 0.0256, 0.0128]
    start=[0., 0.0, -0.0128]
    chrest_data = ChrestData()
    chrest_data.setup_new_grid(start, end, delta)
    
    if args.filerange is None:
        filerange=[0,len(vtcp_data.times)]
        startind=0
    else:
        filerange=args.filerange
        startind=filerange[0]
        
    if args.batchsize is not None:
        if len(vtcp_data.times) > args.batchsize:
            vtcp_data.sort_time(args.batchsize,filerange)
            print("The code processes " + str(args.batchsize) + " files at a time.")
        else:
            vtcp_data.sort_time(len(vtcp_data.times),filerange)
            print("The code processes " + str(len(vtcp_data.times)) + " files at a time.")
    else:
        vtcp_data.sort_time(len(vtcp_data.times),filerange)
        
    import time
    newdir = args.file.parent / (str(args.file.stem).replace("*", "") + ".chrest.vtcp")
    newdir.mkdir(parents=True, exist_ok=True)
    for i in range(0,vtcp_data.numintervals):
        # map the ablate data to chrest
        
        start_time = time.time()
        vtcp_data.convertfield(chrest_data, field_mappings,i, component_select_names, args.max_distance)
        vtcp_data.trace_rays(chrest_data)
        vtcp_data.get_temperature(newdir,i, len(vtcp_data.timeitervals[0]), startind)
        # vtcp_data.get_image()
        print("--- %s seconds ---" % (time.time() - start_time))
        
        # # write the new file without wild card
        # chrest_data_path_base = args.file.parent / (str(args.file.stem).replace("*", "") + ".chrest")
        # chrest_data_path_base.mkdir(parents=True, exist_ok=True)
        # chrest_data_path_base = chrest_data_path_base / (str(args.file.stem).replace("*", "") + ".chrest")

        # # Save the result data
        # chrest_data.savepart(chrest_data_path_base, i, len(vtcp_data.timeitervals[0]), startind)


    # xdmf_file =  newdir / (str(args.file.stem.replace('*', '') + ".xdmf"))

    # hdf5_paths = expand_path(newdir / os.path.basename(str(args.file)))

    # # generate an xdfm object
    # xdfm = XdmfGenerator()

    # # #convert with new path
    # for hdf5_file in hdf5_paths:
    #     # create component markdown
    #     xdfm.append_chrest_hdf5(hdf5_file)

    # # write the xdmf file
    # xdfm.write_to_file(xdmf_file)
    
