# Copyright (c) CryoFold Team, and its affiliates. All Rights Reserved

import os,math
import numpy as np
import torch
from copy import deepcopy
from scipy import ndimage as nd
import time, datetime
import mrcfile

def xcorr(x, y, normed=True, detrend=False, maxlags=None):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    # if detrend:
    #     import matplotlib.mlab as mlab
    #     x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
    #     y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y)

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    # if maxlags is None:
    #     maxlags = Nx - 1

    # if maxlags >= Nx or maxlags < 1:
    #     raise ValueError('maglags must be None or strictly '
    #                      'positive < %d' % Nx)

    # lags = np.arange(-maxlags, maxlags + 1)
    # c = c[Nx - 1 - maxlags:Nx + maxlags]
    # return lags, c
    return c[0]

def mol_atom_density_np(atom_coords, atom_weight, res=3.0, voxel_size=1.0):
    """
    numpy version mol density by coords

    args:
        atom_coords: (N,3)
    """
    if isinstance(atom_coords, torch.Tensor):
        atom_coords = atom_coords.numpy()
    if isinstance(atom_weight, float):
        atom_weight = np.zeros(atom_coords.shape[0]) + atom_weight
    
    bmin = np.floor(atom_coords.min(axis=0)) 
    bmax = np.ceil(atom_coords.max(axis=0))  

    # transform coords into density space
    coords_den = atom_coords / voxel_size
    bcen = (bmax + bmin) / 2 / voxel_size
    
    # mol density
    step_size = 50
    grid_edg  = math.sqrt(-math.log(1e-7))
    gauss_radius = res / math.pi / voxel_size
    gbox = int(grid_edg * gauss_radius)
    grid_size = (gbox+1) * step_size
    gauss_grid = np.exp(-(np.arange(grid_size) / (gauss_radius * step_size))**2)

    def mol_atom(x, y, z):

        ii,jj,kk = [],[],[]
        for k in range(-gbox, gbox):
            ind = int(np.fabs(k - z) * step_size)
            # if ind >=300:
            #     import pdb;pdb.set_trace()
            kk.append(gauss_grid[ind])
        for j in range(-gbox, gbox):
            ind = int(np.fabs(j - y) * step_size)
            # if ind >=300:
            #     import pdb;pdb.set_trace()
            jj.append(gauss_grid[ind])
        for i in range(-gbox, gbox):
            ind = int(np.fabs(i - x) * step_size)
            # if ind >=300:
            #     import pdb;pdb.set_trace()
            ii.append(gauss_grid[ind])
        ade = np.einsum('i,j,k',ii,jj,kk)
        return ade

    boxsize = np.ceil((bmax - bmin) / voxel_size + 2 * gbox).astype(int)
    box0 = (bcen - boxsize / 2).astype(int)
    box1 = (bcen + boxsize / 2).astype(int)
    density = np.zeros((boxsize[0] , boxsize[1] , boxsize[2]))

    for s in range(0, coords_den.shape[0]):
        c = coords_den[s]
        aw = atom_weight[s]
        p0 = (c - gbox).astype(int) - box0
        p1 = (c + gbox).astype(int) - box0
        c  = np.round(c,8)
        atom_cube = aw * mol_atom(c[0]-int(c[0]), c[1]-int(c[1]) ,c[2]-int(c[2]))
        w,h,l = atom_cube.shape
        density[p0[0]:p0[0]+w, p0[1]:p0[1]+h, p0[2]:p0[2]+l] += atom_cube
    return density, box0

def mol_atom_density_th(atom_coords, atom_weight, res=3.0, voxel_size=1.0):
    """
    troch version mol density by coords

    args:
        atom_coords: (N,3)
    """
    if isinstance(atom_coords, np.ndarray):
        atom_coords = torch.froom_numpy(atom_coords)
    if isinstance(atom_weight, float):
        atom_weight = torch.zeros(atom_coords.shape[0]) + atom_weight

    device = atom_coords.device
    bmin = torch.floor(atom_coords.amin(dim=0))
    bmax = torch.ceil(atom_coords.amax(dim=0))
    coords_den = atom_coords / voxel_size
    
    bcen = (bmax + bmin) / 2 / voxel_size
    
    # mol density
    step_size = 50
    grid_edg  = math.sqrt(-math.log(1e-7))
    gauss_radius = res / math.pi / voxel_size
    gbox = int(grid_edg * gauss_radius)
    grid_size = (gbox+1) * step_size
    gsphere = int(4 * gauss_radius)
    def mol_atom(x, y, z):
        ra = torch.arange(-gsphere, gsphere, device=device).float()
        ii = torch.exp(-torch.pow(ra - x.repeat(len(ra)), 2))
        jj = torch.exp(-torch.pow(ra - y.repeat(len(ra)), 2))
        kk = torch.exp(-torch.pow(ra - z.repeat(len(ra)), 2))
        ade = torch.einsum('i,j,k',ii,jj,kk)
        return ade

    boxsize = torch.ceil((bmax - bmin) / voxel_size + 2 * gsphere).int() 
    box0 = torch.floor(bcen - (boxsize.float() / 2.0)).int()
    box1 = torch.floor(bcen + (boxsize.float() / 2.0)).int()
    density = torch.zeros((boxsize[0].item(), boxsize[1].item(), boxsize[2].item()), device=device)
    
    for s in range(0, coords_den.shape[0]):
        c = coords_den[s]
        aw = atom_weight[s]
        
        p0 = (c - gsphere).int() - box0
        atom_cube = aw * mol_atom(c[0]-int(c[0]), c[1]-int(c[1]) ,c[2]-int(c[2]))
        w,h,l = atom_cube.shape
        d_sz = density[p0[0]:p0[0]+w, p0[1]:p0[1]+h, p0[2]:p0[2]+l].shape
        density[p0[0]:p0[0]+w, p0[1]:p0[1]+h, p0[2]:p0[2]+l] += atom_cube[:d_sz[0],:d_sz[1],:d_sz[2]]
        # except:
        #     print(atom_cube.shape)
        #     print(p0[0],p0[0]+w, p0[1],p0[1]+h, p0[2],p0[2]+l)
        #     import pdb;pdb.set_trace()
    
    return density, box0


def mol_atom_density(atom_coords, atom_weight, res=3.0, voxel_size=1.0, datatype="numpy"):
    if datatype=="numpy":
        return mol_atom_density_np(atom_coords, atom_weight, res, voxel_size)
    elif datatype=="torch":
        return mol_atom_density_th(atom_coords, atom_weight, res, voxel_size)
    else:
        raise "wrong datatype: "+datatype

class DensityInfo:
    def __init__(self, mrc_path=None, density=None, offset=(0.0, 0.0, 0.0), apix=1.0, ispg=1, parser="mrc",
                datatype="torch", device=torch.device("cpu"), stats=True, verbose=0):
        self.device = device
        self.datatype = datatype
        self.path = mrc_path
        self.parser = parser
        self.verbose = verbose
        if mrc_path is None:
            self.set_density(density)
            self.set_offset(offset)
            self.set_apix(apix)
            self.set_ispg(ispg)
        else:
            den_info = self.__get_map_data__(mrc_path)
            self.set_density(den_info['density'])
            self.set_offset(den_info['offset'])
            self.set_apix(den_info['apix'])
            self.set_ispg(den_info['ispg'])
        if stats:
            self.set_stats()
        self.shape   = self.density.shape

    def __get_map_data__(self, mrc_path):
        
        if mrc_path.endswith(".npz"):
            if os.path.exists(mrc_path):
                return self.__get_map_data_from_npz__(mrc_path)
            else:
                mrc_path = mrc_path.replace(".npz",".mrc")
        assert os.path.exists(mrc_path), mrc_path+" not found!"

        
        # parse map file
        if self.parser=="mrc":
            return self.__get_map_data_by_mrc__(mrc_path)
        
        try:
            mapfile = mrcfile.open(mrc_path)
        except Exception as e:
            raise (mrc_path, " ERROR!!!!", e)

        dsize = mapfile.data.shape
        if len(dsize) != 3:
            print(mrc_path, "size error:", len(dsize))
            return
        x = int(mapfile.header["nxstart"])
        y = int(mapfile.header["nystart"])
        z = int(mapfile.header["nzstart"])
        c = mapfile.header.mapc - 1
        r = mapfile.header.mapr - 1
        s = mapfile.header.maps - 1
        crs = [c, r, s]
        x0, y0, z0 = list(mapfile.header["origin"].tolist())
        apix = mapfile.voxel_size.x

        offset = [x + x0/apix, y + y0/apix, z + z0/apix]
        offset = (offset[crs[0]], offset[crs[1]], offset[crs[2]])
        
        cella = np.array(mapfile.header["cella"].tolist())
        

        m_data = mapfile.data.T.transpose(crs[0],crs[1],crs[2])
        return {'density': m_data, 'offset': offset, 'apix': apix, 'ispg': 1}

    def __get_map_data_by_mrc__(self, mrc_path):
        from .mrc import MRC
        mapfile = MRC(mrc_path)
        return {'density': mapfile.data, 'offset': mapfile.offset, 'apix': mapfile.apix, 'ispg': 1}

    def set_density(self, density):
        if self.datatype == "numpy":
            self.density = density
        else:
            if isinstance(density, torch.Tensor):
                self.density = density
            elif isinstance(density, np.ndarray):
                self.density = torch.from_numpy(density).to(self.device)
                # self.density = torch.from_numpy(deepcopy(density)).to(self.device)
            else:
                raise "error data type of density {}".format(type(density))
    
    def get_density(self):
        return self.density

    def to(self, device):
        return DensityInfo(density=self.density.to(device), offset=self.offset.to(device), apix=self.apix, device=device)

    def to_torch(self):
        self.datatype = "torch"
        self.density = torch.from_numpy(self.density).to(self.device)
        self.set_offset(self.offset)


    def to_dict(self):
        return dict(
            density=self.density,
            densityinfo=self.get_info(),
            apix=self.apix,
            offset=self.offset)

    def __repr__(self):
        """Print Density object as ."""
        s = self.size
        info = "<DensityInfo ({},{},{}) > \n".format(s[0].item(), s[1].item(), s[2].item(), ) \
             + "offset:  \t({:.4f},{:.4f},{:.4f})\n".format(self.offset[0], self.offset[1], self.offset[2]) \
             + "apix:    \t{:.4f}\n".format(self.apix) \
             + "min:     \t{:.4f}\n".format(self.min) \
             + "max:     \t{:.4f}\n".format(self.max) \
             + "mean:    \t{:.4f}\n".format(self.mean) \
             + "mean1s:  \t{:.4f}\n".format(self.mean1s) \
             + "meanp:   \t{:.4f}\n".format(self.meanp) \
             + "meanp1s: \t{:.4f}\n".format(self.meanp1s) 
        return info

    def set_stats(self):
        self.mean  = self.density.mean()
        self.meanp = self.density[self.density>self.mean].mean()
        
        self.std   = self.density.std()
        self.stdp  = self.density[self.density>self.mean].std()

        self.mean1s = self.mean + self.std
        self.meanp1s = self.meanp + self.stdp

        self.min   = self.density.min()
        self.max   = self.density.max()

        self.size = torch.tensor(self.density.shape, device=self.device)

        self.shape   = self.density.shape
        # self.size    = self.density.shape


    def meanns(self, n=1):
        return self.mean + n*self.std

    def meanpns(self, n=1):
        return self.meanp + n*self.stdp

    

    def get_density_at_ast(self, p):
        dp = torch.round(p/self.apix - self.offset).long()
        return self.get_density_at(dp)
        
    def get_density_at(self, dp):
        if (self.size - 1 - dp).min()>=0:
            return self.density[dp[0], dp[1], dp[2]]
        else:
            return None

    

    def set_offset(self, offset):
        if self.datatype == "numpy":
            if isinstance(offset, tuple) or isinstance(offset, list):
                self.offset = np.array(offset)
            else:
                self.offset = offset
        else:
            if isinstance(offset, torch.Tensor):
                self.offset = offset
            elif isinstance(offset, np.ndarray) or isinstance(offset, tuple) or isinstance(offset, list):
                self.offset = torch.tensor(offset).to(self.device)
            else:
                raise "error data type of offset {}".format(type(offset))
                # self.offset=self.offset.to(torch.float32)

    def get_offset(self):
        return self.offset

    def set_apix(self, apix):
        self.apix = apix
        self.voxel_size = apix

    def get_apix(self):
        return self.apix

    def set_ispg(self, ispg):
        self.ispg = ispg

    def get_ispg(self):
        return self.ispg

    def get_info(self):
        return np.array(
            [
                self.density.shape[0],
                self.density.shape[1],
                self.density.shape[2],
                self.apix,
                self.apix,
                self.apix,
                self.offset[0],
                self.offset[1],
                self.offset[2],
            ]
        )

    def is_empty(self, density):
        if np.isnan(density).any():
            return True
        elif density.max() == density.min() or np.isclose(density,0.0).all():
            return True
        else:
            return False

    def norm(self):
        # norm_log_mean_2s:
        # TODO: norm
        # data = deepcopy(density)
        data = self.density
        mean = np.mean(data)
        # std  = np.std(data)
        data[data<=mean] = 0.0

        nonzero_mean = np.nanmean(np.where(np.isclose(data,0.0), np.nan, data)) 
        nonzero_std  = np.nanstd(np.where(np.isclose(data,0.0), np.nan, data)) 
        
        data = data / (nonzero_mean + 2*nonzero_std)
        if np.isnan(data).any():
            return 
        tmp = data[ (data<1.0) & (data>0) ]
        data[ (data<1.0) & (data>0) ] = (tmp - tmp.min())/(tmp.max() - tmp.min()) 
        data[data>1.0] = 1 + (1/(1+np.exp(-data[data>1.0]+1)) - 0.5)*2
        self.density = data
    
    def pad(self, cube_width, return_mask=False, no_padding=False, scale_factor=4):
        target_shape = list(self.density.shape)
        for i in range(len(target_shape)):
            if no_padding:
                target_shape[i] = target_shape[i] + (scale_factor - target_shape[i] % scale_factor)
            else:
                if target_shape[i] < cube_width:
                    target_shape[i] = cube_width

        value = self.density.min()
        cubes = self.density
        if isinstance(cubes, np.ndarray):
            m = np.array(target_shape) - cubes.shape
            mask = np.ones(target_shape)
            if m[0] > 0:
                b = np.zeros([m[0], cubes.shape[1], cubes.shape[2]]) + value
                mask[cubes.shape[0]:,:,:] = 0
                cubes = np.concatenate((cubes, b), axis=0)
            if m[1] > 0:
                b = np.zeros([cubes.shape[0], m[1], cubes.shape[2]]) + value
                mask[:,cubes.shape[1]:,:] = 0
                cubes = np.concatenate((cubes, b), axis=1)
            if m[2] > 0:
                b = np.zeros([cubes.shape[0], cubes.shape[1], m[2]]) + value
                mask[:,:,cubes.shape[2]:] = 0
                cubes = np.concatenate((cubes, b), axis=2)
        elif isinstance(cubes, torch.Tensor):
            m = (torch.tensor(target_shape) - torch.tensor(cubes.shape)).int()
            mask = torch.ones(target_shape)
            if m[0] > 0:
                b = torch.zeros([m[0], cubes.shape[1], cubes.shape[2]]) + value
                mask[cubes.shape[0]:,:,:] = 0
                cubes = torch.cat((cubes, b), dim=0)
            if m[1] > 0:
                b = torch.zeros([cubes.shape[0], m[1], cubes.shape[2]]) + value
                mask[:,cubes.shape[1]:,:] = 0
                cubes = torch.cat((cubes, b), dim=1)
            if m[2] > 0:
                b = torch.zeros([cubes.shape[0], cubes.shape[1], m[2]]) + value
                mask[:,:,cubes.shape[2]:] = 0
                cubes = torch.cat((cubes, b), dim=2)
        else:
            raise "wrong data type"
        if return_mask:
            return cubes, mask
        else:
            return cubes
        
    def clip0(self, clip_ratio=0.01 , clip_thr=0.0):
        # if pdb_path is not None and os.path.exists(pdb_path):
        #     b_min,b_max = get_box(pdb_path, isreal, offset, voxel_size)
        #     # suffix += "_clip"
        #     print("Cliping `empty` region {} - {} by PDB file {}".format(b_min,b_max, pdb_path))
        # else:
        data = self.density
        dsize = list(data.shape)
        if clip_thr == 0.0:
            thr = data.mean()+data.std()
        else:
            thr = clip_thr
        aa = [ ((data[i]>thr).sum()< clip_ratio*data[i].numel()).item() for i in range(data.shape[0]) ]
        bb = [ ((data[:,i,:]>thr).sum()< clip_ratio*data[:,i,:].numel()).item() for i in range(data.shape[1]) ]
        cc = [ ((data[:,:,i]>thr).sum()< clip_ratio*data[:,:,i].numel()).item() for i in range(data.shape[2]) ]
        print(aa)
        print(bb)
        print(cc)
        # import pdb; pdb.set_trace()
        b_min = [0,0,0]
        b_max = list(data.shape)
        try:
            b_min[0] = aa.index(0)
        except:
            print("all z !!!")
        b_max[0] = data.shape[0] - (sum(aa)-b_min[0])

        try:
            b_min[1] = bb.index(0)
        except:
            print("all z !!!")
        b_max[1] = data.shape[1] - (sum(bb)-b_min[1])
        try:
            b_min[2] = cc.index(0)
        except:
            print("all z !!!")
        b_max[2] = data.shape[2] - (sum(cc)-b_min[2])
        # print("Cliping `empty` region {} - {} by estimation... ".format(b_min,b_max))
        x0,x1 = max(0, b_min[0]-3), min(dsize[0], b_max[0]+3)
        y0,y1 = max(0, b_min[1]-3), min(dsize[1], b_max[1]+3)
        z0,z1 = max(0, b_min[2]-3), min(dsize[2], b_max[2]+3)
        
        self.density = data[x0:x1, y0:y1, z0:z1]
        
        self.offset[0] = self.offset[0]+x0
        self.offset[1] = self.offset[1]+y0
        self.offset[2] = self.offset[2]+z0
    
    def __clip__(self, clip_ratio=0.001 , clip_thr=0.0):
        # if pdb_path is not None and os.path.exists(pdb_path):
        #     b_min,b_max = get_box(pdb_path, isreal, offset, voxel_size)
        #     # suffix += "_clip"
        #     print("Cliping `empty` region {} - {} by PDB file {}".format(b_min,b_max, pdb_path))
        # else:
        data = deepcopy(self.density)
        dsize = list(data.shape)
        if clip_thr == 0.0:
            # thr = data.mean()#+data.std()
            thr = self.meanp1s #+data.std()
            # print("use mean1s")
        else:
            thr = clip_thr
        if isinstance(data, torch.Tensor):
            aa = [ ((data[i]>thr).sum()< clip_ratio*data[i].numel()).item() for i in range(data.shape[0]) ]
            bb = [ ((data[:,i,:]>thr).sum()< clip_ratio*data[:,i,:].numel()).item() for i in range(data.shape[1]) ]
            cc = [ ((data[:,:,i]>thr).sum()< clip_ratio*data[:,:,i].numel()).item() for i in range(data.shape[2]) ]
        else:
            aa = [ (data[i]>thr).sum()< clip_ratio*data[i].size for i in range(data.shape[0]) ]
            bb = [ (data[:,i,:]>thr).sum()< clip_ratio*data[:,i,:].size for i in range(data.shape[1]) ]
            cc = [ (data[:,:,i]>thr).sum()< clip_ratio*data[:,:,i].size for i in range(data.shape[2]) ]
        if self.verbose>0:
            print(aa)
            print(bb)
            print(cc)
        # import pdb; pdb.set_trace()
        b_min = [0,0,0]
        b_max = list(data.shape)
        try:
            b_min[0] = aa.index(0)
        except:
            print("all x !!!")
        b_max[0] = data.shape[0] - (sum(aa)-b_min[0])

        try:
            b_min[1] = bb.index(0)
        except:
            print("all y !!!")
        b_max[1] = data.shape[1] - (sum(bb)-b_min[1])
        try:
            b_min[2] = cc.index(0)
        except:
            print("all z !!!")
        b_max[2] = data.shape[2] - (sum(cc)-b_min[2])
        # print("Cliping `empty` region {} - {} by estimation... ".format(b_min,b_max))
        x0,x1 = max(0, b_min[0]-3), min(dsize[0], b_max[0]+3)
        y0,y1 = max(0, b_min[1]-3), min(dsize[1], b_max[1]+3)
        z0,z1 = max(0, b_min[2]-3), min(dsize[2], b_max[2]+3)
        
        density = data[x0:x1, y0:y1, z0:z1]

        offset = deepcopy(self.offset)
        offset[0] = self.offset[0]+x0
        offset[1] = self.offset[1]+y0
        offset[2] = self.offset[2]+z0
        return density, offset

    def __clip__mask(self, msk_den):
        # if pdb_path is not None and os.path.exists(pdb_path):
        #     b_min,b_max = get_box(pdb_path, isreal, offset, voxel_size)
        #     # suffix += "_clip"
        #     print("Cliping `empty` region {} - {} by PDB file {}".format(b_min,b_max, pdb_path))
        # else:
        data = deepcopy(msk_den.density)
        dsize = list(data.shape)
        
        thr = 1
        if isinstance(data, torch.Tensor):
            aa = [ ((data[i]>=thr).sum()==0).item() for i in range(data.shape[0]) ]
            bb = [ ((data[:,i,:]>=thr).sum()==0).item() for i in range(data.shape[1]) ]
            cc = [ ((data[:,:,i]>=thr).sum()==0).item() for i in range(data.shape[2]) ]
        else:
            aa = [ (data[i]>=thr).sum()==0 for i in range(data.shape[0]) ]
            bb = [ (data[:,i,:]>=thr).sum() ==0 for i in range(data.shape[1]) ]
            cc = [ (data[:,:,i]>=thr).sum() ==0 for i in range(data.shape[2]) ]
        if self.verbose>0:
            print(aa)
            print(bb)
            print(cc)
        # import pdb; pdb.set_trace()
        b_min = [0,0,0]
        b_max = list(data.shape)
        try:
            b_min[0] = aa.index(0)
        except:
            print("all x !!!")
        b_max[0] = data.shape[0] - (sum(aa)-b_min[0])

        try:
            b_min[1] = bb.index(0)
        except:
            print("all y !!!")
        b_max[1] = data.shape[1] - (sum(bb)-b_min[1])
        try:
            b_min[2] = cc.index(0)
        except:
            print("all z !!!")
        b_max[2] = data.shape[2] - (sum(cc)-b_min[2])
        # print("Cliping `empty` region {} - {} by estimation... ".format(b_min,b_max))
        x0,x1 = max(0, b_min[0]-3), min(dsize[0], b_max[0]+3)
        y0,y1 = max(0, b_min[1]-3), min(dsize[1], b_max[1]+3)
        z0,z1 = max(0, b_min[2]-3), min(dsize[2], b_max[2]+3)
        
        density = self.density[x0:x1, y0:y1, z0:z1]

        offset = deepcopy(self.offset)
        offset[0] = self.offset[0]+x0
        offset[1] = self.offset[1]+y0
        offset[2] = self.offset[2]+z0
        return density, offset
    
    def clip(self, clip_ratio=0.01 , clip_thr=0.0, use_mask=None):
        if use_mask is not None:
            density, offset = self.__clip__mask(use_mask)
        else:
            density, offset = self.__clip__(clip_ratio=clip_ratio , clip_thr=clip_thr)
            # import pdb;pdb.set_trace()
            times = 0
            while self.is_empty(density) and times<3:
                print("clip into empty density, clip_ratio: {} ({}).".format(clip_ratio, times))
                clip_ratio = clip_ratio*0.1
                density, offset = self.__clip__(clip_ratio=clip_ratio , clip_thr=clip_thr)
                times += 1
        self.density = density
        self.offset = offset
        self.set_stats()

    def scale(self, apix=1.0):
        scale_ratio = self.apix / apix
        if not np.allclose(scale_ratio, 1.0):
            if isinstance(self.density, torch.Tensor):
                # from torch.nn.functional import interpolate
                # new_density = interpolate(self.density, scale_factor=scale_ratio, mode='linear')
                offset = self.offset
                if self.density.is_cuda:
                    density = self.density.cpu().numpy()
                    offset  = self.offset.cpu().numpy()
                else:
                    density = self.density.numpy()
                    offset  = self.offset.numpy()
                new_density = nd.interpolation.zoom(density, scale_ratio)
            elif isinstance(self.density, np.ndarray):
                new_density = nd.interpolation.zoom(self.density, scale_ratio)
                offset = self.offset
            self.set_density(new_density)
            self.set_offset(offset * scale_ratio)
        self.set_apix(apix)
        self.set_stats()


    def cos_sim(self, mol_den):
        t_ov, m_ov = self.overlap_right(mol_den)
        cs = torch.nn.functional.cosine_similarity(t_ov.reshape(1,-1), m_ov.reshape(1,-1))
        return cs

    def cross_corr(self, mol_den):
        t_ov, m_ov = self.overlap_right(mol_den)
        cc = xcorr(t_ov.reshape(-1).numpy(), m_ov.reshape(-1).numpy())
        return cc

    def overlap(self, mol_den):
        tgt_den = self
        device  = tgt_den.device
        tgt_off = tgt_den.get_offset()
        mol_off = mol_den.get_offset()
        t_s = tgt_den.shape
        m_s = mol_den.shape

        t_whl = torch.tensor(t_s, device=device)
        m_whl = torch.tensor(m_s, device=device)
        t_box = torch.cat((tgt_off, tgt_off + t_whl.float())).int()
        m_box = torch.cat((mol_off, mol_off + m_whl.float())).int()

        bb     = torch.stack((t_box,m_box))
        bmin,_ = bb[:,:3].min(dim=0)
        bmax,_ = bb[:,3:].max(dim=0)
        nwhl   = bmax - bmin + 1

        t_o = t_box[:3] - bmin
        m_o = m_box[:3] - bmin

        t_e = t_o.long() + t_whl.long()
        m_e = m_o.long() + m_whl.long()

        t_ov = torch.zeros((nwhl[0], nwhl[1], nwhl[2]), device=device)
        m_ov = torch.zeros((nwhl[0], nwhl[1], nwhl[2]), device=device)

        t_ov[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] = tgt_den.density
        m_ov[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]] = mol_den.density
        
        return t_ov.reshape(1,-1), m_ov.reshape(1,-1)

    def overlap_right(self, mol_den):
        tgt_den = self
        device  = tgt_den.device
        tgt_off = tgt_den.get_offset().int()
        mol_off = mol_den.get_offset().int()
        t_s = tgt_den.shape
        m_s = mol_den.shape

        t_whl = torch.tensor(t_s, device=device).int()
        m_whl = torch.tensor(m_s, device=device).int()

        t_box = torch.cat((tgt_off, tgt_off + t_whl))
        m_box = torch.cat((mol_off, mol_off + m_whl))
        
        bb     = torch.stack((t_box,m_box))
        bl,_ = bb[:,:3].max(dim=0)
        br,_ = bb[:,3:].min(dim=0)
        nwhl   = br - bl 

        t_o = bl - tgt_off
        m_o = bl - mol_off

        t_e = t_o + nwhl
        m_e = m_o + nwhl

        t_ov = torch.zeros_like(mol_den.density, device=device)
        # import pdb;pdb.set_trace()
        s0 = t_ov[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]].shape
        s1 = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]].shape
        if s0 == s1:
            t_ov[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]] = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] 
            m_ov = mol_den.density.reshape(-1)
            t_ov = t_ov.reshape(-1)[m_ov>0]
            m_ov = m_ov[m_ov>0]
            return t_ov.reshape(1,-1), m_ov.reshape(1,-1)
        else:
            print("warning!!! two different shape: ", s0, s1)
            return torch.zeros(2,2).reshape(1,-1), torch.ones(2,2).reshape(1,-1)

    def overlap_left(self, mol_den):
        tgt_den = self
        device  = tgt_den.device
        tgt_off = tgt_den.get_offset().int()
        mol_off = mol_den.get_offset().int()
        t_s = tgt_den.shape
        m_s = mol_den.shape

        t_whl = torch.tensor(t_s, device=device)
        m_whl = torch.tensor(m_s, device=device)

        t_box = torch.cat((tgt_off, tgt_off + t_whl))
        m_box = torch.cat((mol_off, mol_off + m_whl))
        
        # bb     = torch.stack((tgt_off, mol_off))
        # bmin,_ = bb[:,:3].min(dim=0)
        bb     = torch.stack((t_box,m_box))
        bl,_ = bb[:,:3].max(dim=0)
        br,_ = bb[:,3:].min(dim=0)
        nwhl   = br - bl 

        # ww     = torch.stack((t_whl, m_whl))
        # wmin,_ = ww[:,:3].min(dim=0)
        # bmax,_ = bb[:,3:].max(dim=0)
        # nwhl   = bmax - bmin + 1

        t_o = bl - tgt_off
        m_o = bl - mol_off

        # t_o = t_box[:3] - bmin
        # m_o = msk_off.int() - tgt_off.int()

        t_e = t_o + nwhl
        m_e = m_o + nwhl

        m_ov = torch.zeros_like(tgt_den.density, device=device)
        # import pdb;pdb.set_trace()
        m_ov[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] = mol_den.density[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]]
        t_ov = tgt_den.density.reshape(-1)
        m_ov = m_ov.reshape(-1)[t_ov>0]
        t_ov = t_ov[t_ov>0]
        return t_ov.reshape(1,-1), m_ov.reshape(1,-1)

    def mask(self, msk_den, keep_origin=False):
        if self.datatype != 'torch':
            self.to_torch()
        tgt_den = self
        device  = tgt_den.device
        tgt_off = tgt_den.get_offset().int()
        msk_off = msk_den.get_offset().int()
        t_s = tgt_den.shape
        m_s = msk_den.shape

        t_whl = torch.tensor(t_s, device=device)
        m_whl = torch.tensor(m_s, device=device)

        bb     = torch.stack((tgt_off, msk_off))
        bmin,_ = bb[:,:3].max(dim=0)

        ww     = torch.stack((t_whl, m_whl))
        bmax,_ = (bb+ww)[:,:3].min(dim=0)
        nwhl   = bmax - bmin

        t_o =  bmin - tgt_off
        m_o =  bmin - msk_off

        t_e = t_o + nwhl 
        m_e = m_o + nwhl 
        
        # what ??????
        tgt_region = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]]
        msk_region = msk_den.density[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]]

        s0 = tgt_region.shape
        s1 = msk_region.shape
        if s0 == s1:
            msk_density = tgt_region * msk_region
            if keep_origin:
                tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] = msk_density
                # import pdb;pdb.set_trace()
                return tgt_den.density, tgt_off
            else:
                return msk_density, bmin
        else:
            return torch.empty(0), bmin
    
    def mask_box(self, box_min, box_max, keep_origin=False):
        if self.datatype != 'torch':
            self.to_torch()
        tgt_den = self
        device  = tgt_den.device
        tgt_off = tgt_den.get_offset().int()
        # msk_off = msk_den.get_offset().int()
        t_s = tgt_den.shape
        # m_s = msk_den.shape

        box_min = box_min.int()
        box_max = box_max.int()

        t_whl = torch.tensor(t_s, device=device)
        m_whl = torch.tensor(box_max-box_min, device=device)

        bb     = torch.stack((tgt_off, box_min))
        bmin,_ = bb[:,:3].max(dim=0)

        ww     = torch.stack((t_whl, m_whl))
        bmax,_ = (bb+ww)[:,:3].min(dim=0)
        nwhl   = bmax - bmin

        t_o =  bmin - tgt_off
        m_o =  bmin - box_min

        t_e = t_o + nwhl 
        m_e = m_o + nwhl 
        
        # what ??????
        tgt_region = tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]]
        # msk_region = msk_den.density[m_o[0]:m_e[0], m_o[1]:m_e[1], m_o[2]:m_e[2]]
        # msk_density = tgt_region * msk_region
        if keep_origin:
            tgt_den.density = torch.zeros_like(tgt_den.density)
            tgt_den.density[t_o[0]:t_e[0], t_o[1]:t_e[1], t_o[2]:t_e[2]] = tgt_region
            # import pdb;pdb.set_trace()
            return tgt_den.density, tgt_off
        else:
            return tgt_region, bmin

    def int(self, var):
        if isinstance(var, torch.Tensor):
            return var.int()
        else:
            return var.astype(int)

    def save(self, map_path):
        if self.parser =="mrc":
            self.save_mrc_new(map_path, self.density, self.offset, self.apix, self.ispg)
        else:
            self.save_mrc(map_path, self.density, self.offset, self.apix, self.ispg)

    @staticmethod
    def save_mrc_new(map_path, data, offset=(0.0, 0.0, 0.0), apix=1.0, ispg=1): 
        from .mrc import write
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                data= data.cpu()
            data = data.detach().numpy()
        if type(data[0,0,0])!=np.float32:
            data = data.astype(np.float32)
        
        if isinstance(offset, torch.Tensor):
            if offset.is_cuda:
                offset = offset.cpu()
            offset = offset.detach().numpy()
        if isinstance(apix, torch.Tensor):
            if apix.is_cuda:
                apix = apix.cpu()
            apix = apix.detach().numpy()
        # elif len(apix)>1:
        #     apix = apix[0]

        write(map_path, data.T, header=None, Apix=apix, nxyzstart=offset)

    @staticmethod
    def save_mrc(map_path, data, offset=(0.0, 0.0, 0.0), apix=1.0, ispg=1): 
        
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                data= data.cpu()
            data = data.detach().numpy()
        if type(data[0,0,0])!=np.float32:
            data = data.astype(np.float32)
        
        if isinstance(offset, torch.Tensor):
            if offset.is_cuda:
                offset = offset.cpu()
            offset = offset.detach().numpy()
        if isinstance(apix, torch.Tensor):
            if apix.is_cuda:
                apix = apix.cpu()
            apix = apix.detach().numpy()[0]
        # print(voxel_size)
        # import pdb;pdb.set_trace()
        new_map = mrcfile.new(map_path, overwrite=True)
        new_map.set_data(data.T)
        new_map.header.nx = data.shape[0]
        new_map.header.ny = data.shape[1]
        new_map.header.nz = data.shape[2]
        new_map.header.nxstart  = offset[0]
        new_map.header.nystart  = offset[1]
        new_map.header.nzstart  = offset[2]
        new_map.header.mapc = 1
        new_map.header.mapr = 2
        new_map.header.maps = 3
        new_map.header.cella.x = data.shape[0] * apix
        new_map.header.cella.y = data.shape[1] * apix
        new_map.header.cella.z = data.shape[2] * apix
        new_map.header.mx = data.shape[0]
        new_map.header.my = data.shape[1]
        new_map.header.mz = data.shape[2]
        new_map.header.ispg = 1
        new_map.header.nversion = 20190801
        new_map.header.label[1] = 'by CryoNet, Author: Kui Xu, xukui.cs@gmail.com, Tsinghua University.'
        new_map.header.label[2] = "{:.6f}, {:.6f}, {:.6f}".format(offset[0], offset[1], offset[2])
        new_map.header.label[3] = "apix: {:.6f}".format(apix)
        new_map.header.label[4] = "MODIFIED: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        new_map.header.nlabl = 5
        new_map.close()
        # print('Map saved into: ',map_path)

