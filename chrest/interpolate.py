import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np
from scipy.interpolate import griddata
import time
import os
import h5py
import csv
from multiprocessing import Pool

def interp(values, vtx, wts, points, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    mask=np.any(wts < 0, axis=1)
    print(values.shape)
    print(points.shape)
    print(ret.shape)

    for i in range(len(ret)):
        if mask[i]:
            ret[i]=values[points[i]]
    # assigning nan for outside points should work but it get sketch around 
    # edges...
    # ret[np.any(wts < -1e10, axis=1)] = fill_value
    return ret

def grad(values, vtx, wts):
    #d_phi/dL1 =phi1 since linear weights are used 
    values=values.reshape((len(values), 1))
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    return ret


def find_simplex_piece(args):
    piece, tri = args
    return tri.find_simplex(piece)

def get_gradweights(vertices):

    # Extract vertex coordinates and temperatures for the current element
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # 3D
    # Calculate the coordinate matrix
    A = np.array([[ 1, 1, 1, 1], [x[0], x[1], x[2], x[3]], [y[0], y[1], y[2], y[3]], [z[0], z[1], z[2], z[3]]])  

    # Solve for natural coordinates -- if the triangulation is messed up eg.:
    # all points are in one plane the determinant is 0... This shouldnt be an 
    # issue since all the points should be contained volumes...
    try:
        invA = np.linalg.inv(A)
    except:
        invA = np.full((4, 4), np.nan)
        # print("Matrix is not invertible!")

    # dL1/dx dL2/dx dL3/dx ... ... dL3/dz dL4/dz
    #this is the format for saving the weights
    return np.array([[invA[0,1], invA[1,1], invA[2,1], invA[3,1]], 
                     [invA[0,2], invA[1,2], invA[2,2], invA[3,2]], 
                     [invA[0,3], invA[1,3], invA[2,3], invA[3,3]]])

def calc_interpweights_3D(xyz,uvw,path):

    
    def interp_weights(xyz, uvw, d=3):
        # start_time = time.time()
        tri = qhull.Delaunay(xyz, qhull_options='QJ')
        # print("--- %s seconds for triangulation ---" % (time.time() - start_time))
                
        from multiprocessing import cpu_count
        n_cores = cpu_count()
        
        pieces = np.array_split(uvw, n_cores)
        # Create a pool of workers
        with Pool(processes=n_cores) as pool:
            results = pool.map(find_simplex_piece, [(piece, tri) for piece in pieces])
        simplex = np.concatenate(results)
        
        # or do it using one node...
        # simplex = tri.find_simplex(uvw)

        #getting the right verticies
        vertices = np.take(tri.simplices, simplex, axis=0)
        
        # mapping triangulation into barycentric coord    
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:, d]

        # normalized barycentric coords
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)

        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))    

    
    filename = 'linweight3D_ablate_'+str(len(xyz))+'chrest_'+str(len(uvw))+'.csv'

    filepath = path / filename
    if not filepath.exists():
        print('Solving for weights')
        vtx, wts = interp_weights(xyz, uvw)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Vertex 1', 'Vertex 2', 'Vertex 3','Vertex 4', 'Weight 1', 'Weight 2', 'Weight 3', 'Weight 4'])
            for i in range(len(vtx)):
                row = list(vtx[i]) + list(wts[i])
                writer.writerow(row)
        print(f'Saved {filename} successfully.')
    else:
        print('Reading in precomputed weights')
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)[1:]  # Skip the header row
            
        vtx = np.array([[int(round(float(row[i]))) for i in range(4)] for row in rows], dtype=np.int64)
        wts = np.array([[float(row[i]) for i in range(4, 8)] for row in rows], dtype=np.float64)
        print(f'Read {filename} successfully.')        
    
    
    return vtx,wts

def calc_allweights_3D(xyz,uvw,path):

    
    def interp_weights_all(xyz, uvw, d=3):
        
        tri = qhull.Delaunay(xyz, qhull_options='QJ')
        
        from multiprocessing import cpu_count
        n_cores = cpu_count()
        
        pieces = np.array_split(uvw, n_cores)
        # Create a pool of workers
        with Pool(processes=n_cores) as pool:
            results = pool.map(find_simplex_piece, [(piece, tri) for piece in pieces])
        simplex = np.concatenate(results)
        
        # or do it using one node...
        # simplex = tri.find_simplex(uvw)

        #getting the right verticies
        vertices = np.take(tri.simplices, simplex, axis=0)
        
        # mapping triangulation into barycentric coord    
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:, d]

        # normalized barycentric coords
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        
        ind=0
        W1=np.zeros_like(tri.simplices, dtype=float)
        W2=np.zeros_like(tri.simplices, dtype=float)
        W3=np.zeros_like(tri.simplices, dtype=float)
        for simp in tri.simplices:
            # Get the vertex indices of the triangle
            W1[ind,:], W2[ind,:], W3[ind,:] = get_gradweights(tri.points[simp])
            ind+=1
        
        Wx=np.take(W1, simplex, axis=0)
        Wy=np.take(W2, simplex, axis=0)
        Wz=np.take(W3, simplex, axis=0)
        
        #setting the values to nan for points that are not in the mesh
        Wx[simplex == -1,:] = np.nan
        Wy[simplex == -1,:] = np.nan
        Wz[simplex == -1,:] = np.nan
        
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True))), Wx, Wy, Wz

    filename = 'allweight3D_ablate_'+str(len(xyz))+'chrest_'+str(len(uvw))+'.csv'
    filepath = path / filename
    
    if not filepath.exists():
        print('Solving for all the weights')
        vtx, wts, Wx, Wy, Wz= interp_weights_all(xyz, uvw)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Vertex 1', 'Vertex 2', 'Vertex 3','Vertex 4', 'Weight 1', 'Weight 2', 'Weight 3', 'Weight 4',
                              'dL1/dx', 'dL2/dx', 'dL3/dx', 'dL4/dx', 'dL1/dy', 'dL2/dy', 'dL3/dy', 'dL4/dy',
                              'dL1/dz', 'dL2/dz', 'dL3/dz', 'dL4/dz'])
            for i in range(len(vtx)):
                row = list(vtx[i]) + list(wts[i]) + list(Wx[i]) + list(Wy[i]) + list(Wz[i])
                writer.writerow(row)
        print(f'Saved {filename} successfully.')
    else:
        print('Reading in precomputed weights')
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)[1:]  # Skip the header row
    
        vtx = np.array([[int(round(float(row[i]))) for i in range(4)] for row in rows], dtype=np.int64)
        wts = np.array([[float(row[i]) for i in range(4, 8)] for row in rows], dtype=np.float64)
        Wx = np.array([[float(row[i]) for i in range(8, 12)] for row in rows], dtype=np.float64)
        Wy = np.array([[float(row[i]) for i in range(12, 16)] for row in rows], dtype=np.float64)
        Wz = np.array([[float(row[i]) for i in range(16, 20)] for row in rows], dtype=np.float64)
        print(f'Read {filename} successfully.')
        
    
    return vtx, wts, Wx, Wy, Wz
        
def interpolate3D(vtx,wts,field,points):       

    valuesi=interp(field, vtx, wts, points)

    return valuesi
