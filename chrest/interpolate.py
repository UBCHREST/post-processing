# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:36:45 2023

@author: rkolo
"""

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
from scipy.interpolate import griddata
import time
import os
import h5py
import csv
from multiprocessing import Pool

def interp(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < -1e10, axis=1)] = fill_value
    return ret

def find_simplex_piece(args):
    piece, tri = args
    return tri.find_simplex(piece)

def calcweight3D(xyz,uvw):

    
    def interp_weights(xyz, uvw, d=3):
        start_time = time.time()
        tri = qhull.Delaunay(xyz)
        print("--- %s seconds for triangulation ---" % (time.time() - start_time))
        
        start_time = time.time()
        
        from multiprocessing import cpu_count
        n_cores = cpu_count()
        
        pieces = np.array_split(uvw, n_cores)
        # Create a pool of workers
        with Pool(processes=n_cores) as pool:
            results = pool.map(find_simplex_piece, [(piece, tri) for piece in pieces])
        simplex = np.concatenate(results)
        print("--- %s seconds for finding simplices containing the points ---" % (time.time() - start_time))
        
        start_time = time.time()
        vertices = np.take(tri.simplices, simplex, axis=0)
        print("--- %s seconds for getting the vertices from triangulation ---" % (time.time() - start_time))
        
        start_time = time.time()    
        temp = np.take(tri.transform, simplex, axis=0)
        print("--- %s seconds mapping triangulation into barycentric coord ---" % (time.time() - start_time))
        
        start_time = time.time()
        delta = uvw - temp[:, d]
        print("--- %s seconds calculating local mapping coordinates ---" % (time.time() - start_time))
        
        start_time = time.time()
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        print("--- %s seconds getting the normalized barycentric coords ---" % (time.time() - start_time))
        
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))    

    start_time = time.time()
    # vtx, wts = interp_weights(xyz, uvww)
    
    filename = 'linweight3D_ablate_'+str(len(xyz))+'chrest_'+str(len(uvw))+'.csv'
    # filename2 = 'verticies_ablate_'+str(len(xyz))+'chrest_'+str(len(uvw))+'.csv'
    if not os.path.exists(filename):
        vtx, wts = interp_weights(xyz, uvw)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Vertex 1', 'Vertex 2', 'Vertex 3','Vertex 4', 'Weight 1', 'Weight 2', 'Weight 3', 'Weight 4'])
            for i in range(len(vtx)):
                row = list(vtx[i]) + list(wts[i])
                writer.writerow(row)
        print(f'Saved {filename} successfully.')
    else:

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)[1:]  # Skip the header row
            
    
        vtx = np.array([[int(round(float(row[i]))) for i in range(4)] for row in rows], dtype=np.int64)
        wts = np.array([[float(row[i]) for i in range(4, 8)] for row in rows], dtype=np.float64)
        print(f'Read {filename} successfully.')
        
    print("--- %s seconds for calculating the weights---" % (time.time() - start_time))
    
    return vtx,wts
        
def interpolate3D(vtx,wts,field):       
        
    start_time = time.time()
    # valuesi = interpolate(values.flatten(), vtx, wts)
    valuesi=interp(field, vtx, wts)
    print("--- %s seconds for interpolation with the precalc weights---" % (time.time() - start_time))
    
    return valuesi
    
    # valuesi=valuesi.reshape(Xi.shape[0],Xi.shape[1],Xi.shape[2])