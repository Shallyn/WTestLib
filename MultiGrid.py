#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:06:37 2019

@author: drizl
"""

import numpy as np
from pathlib import Path
import os

# 1D
class Grid1DAtom(object):
    def __init__(self, N_x):
        if N_x <= 0:
            raise Exception(f'Negative N = {N_x}')
        self._N_x = N_x

    @property
    def N_x(self):
        return self._N_x

    @property
    def dxAtom(self):
        return 1 / (self._N_x + 1)

    @property
    def xAtom(self):
        return np.asarray([(i+1)*self.dxAtom for i in range(self._N_x)])

    @property
    def length(self):
        return int(self._N_x)

    def __getitem__(self, key):
        return self.xAtom[key]

class Grid1D(Grid1DAtom):
    def __init__(self, x_min, x_max, N_x, family = '0'):
        super(Grid1D, self).__init__(N_x)
        if x_min >= x_max:
            raise Exception(f'Range incorrect: x_min = {x_min}, x_max = {x_max}')
        self._x_min = x_min
        self._x_max = x_max
        self._family = family

    @property
    def family(self):
        return self._family

    @property
    def xlen(self):
        return self._x_max - self._x_min

    @property
    def dx(self):
        return self.xlen * self.dxAtom

    @property
    def x(self):
        return self._x_min + (self._x_max - self._x_min) * self.xAtom

    def __getitem__(self, key):
        return self.x[key]
    
    def __call__(self, func, **kwargs):
        x_list = self.x
        fx_list = np.zeros(self.length)
        for i in range(self.length):
            fx_list[i] = func(x_list[i], **kwargs)
        return Grid1DFunc(fx_list, self)


class Grid1DFunc(object):
    def __init__(self, fx, grid):
        self._grid = grid
        self._values = np.asarray(fx)
    
    @property
    def grid(self):
        return self._grid

    @property
    def values(self):
        return self._values

    @property
    def value(self):
        return self._values

    @property
    def length(self):
        return self._grid.length

    def __getitem__(self, key):
        if isinstance(key, tuple):
            raise Exception(f'Incorporate slice {key}')
        return self._grid.x[key], self._values[key]

    def calculate_divergence_from_index(self, index):
        ix = int(index)
        val = self._values
        dx = self._grid.dx
        # df / dx
        if ix == 0:
            dfdx = (val[ix + 1] - val[ix]) / dx
        elif ix == self._grid.N_x - 1:
            dfdx = (val[ix] - val[ix-1]) / dx
        else:
            dfdx = (val[ix+1] - val[ix-1]) / dx / 2
        return dfdx

    def filter(self, thresh = 0.5):
        isorted = np.argsort(self._values)[::-1]
        ithresh = int(thresh * self.length)
        icollect = isorted[:ithresh]
        return Grid1DFuncPointsCollector(self, icollect, family = self._grid.family)

    def save(self, fname):
        fpath = Path(fname)
        x = self._grid.x
        val = self._values
        if fpath.exists():
            with open(fpath, 'a') as f:
                for i in range(len(x)):
                    f.write("%.16e %.16e\n"%(x[i], val[i]))
        else:
            np.savetxt(fpath, np.stack([x,val], axis = 1))
        return

    def save_family(self, prefix):
        prefix = Path(prefix)
        fname = prefix / f'grid_{self._grid.family}.dat'
        x = self._grid.x
        val = self._values
        np.savetxt(fname, np.stack([x,val], axis = 1))
        return

# Grid points collect
class Grid1DFuncPointsCollector(object):
    def __init__(self, gridfunc, indexes, family = '0'):
        self._gridfunc = gridfunc
        self._indexes = indexes
        self._family = family

    @property
    def gridfunc(self):
        return self._gridfunc

    def get_points(self):
        return self._gridfunc.grid.x[self._indexes], self._gridfunc.values[self._indexes]

    def __iter__(self):
        xR = self._gridfunc.grid.x
        for ind in self._indexes:
            yield ind, xR[ind], self._gridfunc.values[ind]

    def iterAtom(self):
        xR = self._gridfunc.grid.xAtom
        for ind in self._indexes:
            yield ind, xR[ind], self._gridfunc.values[ind]

        
    # DBSCN cluster
    # Input collector of [(x,...), (y,...), (fxy,...)], and thresh epsilon
    # Return collector of indexes [(ind,...), (ind,...),...]
    def DBSCAN_cluster(self, eps = 1.1):
        CluIndOut = []
        eps = self._gridfunc.grid.dxAtom * eps
        unvisited = self._indexes.copy().tolist()
        visited = []
        xRA = self._gridfunc.grid.xAtom
        while len(unvisited) > 0:
            ip = np.random.choice(unvisited)
            unvisited.remove(ip)
            visited.append(ip)
            xp = xRA[ip]
            CluNew = []
            for ind, x, _ in self.iterAtom():
                if np.abs(x - xp) < eps:
                    CluNew.append(ind)
            iiClu = 0
            while iiClu < len(CluNew):
                ipi = CluNew[iiClu]
                if ipi in unvisited:
                    unvisited.remove(ipi)
                    visited.append(ipi)
                    xi = xRA[ipi]
                    for ind, x, _ in self.iterAtom():
                        if ind not in CluNew and np.abs(x - xi) < eps:
                            CluNew.append(ind)
                iiClu += 1
            CluIndOut.append(CluNew)
        ret = []
        for i, idx_list in enumerate(CluIndOut):
            ret.append(Grid1DFuncPointsCollector(self._gridfunc, np.asarray(idx_list), family = f'{self._family}_{i}'))
        return ret

    def generate_grid(self, N_x = None):
        xp, _ = self.get_points()
        dx = self._gridfunc.grid.dx
        x_min = np.min(xp) - dx
        x_max = np.max(xp) + dx
        if N_x is None:
            dx_new = dx / 5
            N_x = int( (x_max - x_min) / dx_new )
        GrdNew = Grid1D(x_min, x_max, N_x, family = self._family)
        return GrdNew
    
# Used when searching for (x,y) that can maximize func(x,y)
class MultiGrid1D(object):
    def __init__(self, func, x_range, N_x, **kwargs):
        x_min = x_range[0]
        x_max = x_range[1]
        self._grid = Grid1D(x_min, x_max, N_x)
        self._func = func
        self._kwargs = kwargs

    @property
    def grid(self):
        return self._grid
    
    def run(self, fsave = None, eps = 1e-6, magnification = 10, filter_thresh = 0.4, maxiter = 100):
        print_family = True
        if fsave is None:
            print_family = False
        GrdF = self._grid(self._func, **self._kwargs)
        if print_family:
            fpath = Path(fsave)
            if fpath.exists():
                f = open(fpath, 'w')
                f.close()
            fprefix_family = fpath.parent / 'grid_family'
            if not fprefix_family.exists():
                fprefix_family.mkdir(parents = True)
            GrdF.save(fsave)
            GrdF.save_family(fprefix_family)
        GPoints = GrdF.filter(thresh=filter_thresh)
        GPointsClusterList = GPoints.DBSCAN_cluster()
        GrdFuncList = []
        ret = np.stack([GrdF.grid.x, GrdF.values], axis = 1)
        for GPCobj in GPointsClusterList:
            Grd = GPCobj.generate_grid()
            GrdF = Grd(self._func, **self._kwargs)
            if print_family:
                GrdF.save(fsave)
                GrdF.save_family(fprefix_family)
            ret = np.append(ret, np.stack([GrdF.grid.x, GrdF.values], axis = 1), axis = 0)
            GrdFuncList.append(GrdF)
        ind = 0
        dx_init = self._grid.dx
        while (len(GrdFuncList) > 0 and ind < maxiter):
            ind = ind + 1
            GrdF = GrdFuncList[0]
            # check program end
            dx = GrdF.grid.dx
            div = GrdF.calculate_divergence_from_index(np.argmax(GrdF.values))
            if np.abs(div) < eps and dx < dx_init/magnification:
                GrdFuncList.remove(GrdF)
                continue
            GPoints = GrdF.filter(thresh = filter_thresh)
            GPointsClusterList = GPoints.DBSCAN_cluster()
            GrdFuncListNew = []
            for GPCobj in GPointsClusterList:
                GrdNew = GPCobj.generate_grid()
                GrdFNew = GrdNew(self._func, **self._kwargs)
                if print_family:
                    GrdFNew.save(fsave)
                    GrdFNew.save_family(fprefix_family)
                ret = np.append(ret, np.stack([GrdFNew.grid.x, GrdFNew.values], axis = 1), axis = 0)
                GrdFuncListNew.append(GrdFNew)
            GrdFuncList.remove(GrdF)
            GrdFuncList = GrdFuncList + GrdFuncListNew
        return ret


# 2D
class Grid2DAtom(object):
    def __init__(self, N_x, N_y):
        if N_x <= 0 or N_y <= 0:
            raise Exception(f'Negative N_x = {N_x} or N_y = {N_y}')
        self._N_x = N_x
        self._N_y = N_y

    @property
    def N_x(self):
        return self._N_x

    @property
    def N_y(self):
        return self._N_y

    @property
    def dxAtom(self):
        return 1 / (self._N_x + 1)

    @property
    def dyAtom(self):
        return 1 / (self._N_y + 1)

    @property
    def dlAtom(self):
        return np.sqrt(self.dxAtom**2 + self.dyAtom**2)

    @property
    def xAtom(self):
        return np.asarray([(i+1)*self.dxAtom for i in range(self._N_x)])

    @property
    def yAtom(self):
        return np.asarray([(i+1)*self.dyAtom for i in range(self._N_y)])

    @property
    def xRankAtom(self):
        return np.asarray(self._N_y * self.xAtom.tolist())

    @property
    def meshgridAtom(self):
        return np.meshgrid(self.xAtom, self.yAtom)

    @property
    def yRankAtom(self):
        yret = []
        yA = self.yAtom
        for i in range(self._N_y):
            yret = yret + self._N_x * [yA[i]]
        return np.asarray(yret)

    @property
    def length(self):
        return int(self._N_x * self._N_y)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) > 2:
                raise Exception(f'Incorporate slice {key}')
            return self.xAtom[key[0]], self.yAtom[key[1]]
        return self.xRankAtom[key], self.yRankAtom[key]

    def index_rank2xy(self, index):
        ix = index%self._N_x
        iy = int(index/self._N_x)
        return ix, iy

class Grid2D(Grid2DAtom):
    def __init__(self, x_min, x_max, y_min, y_max, N_x, N_y, family = '0'):
        super(Grid2D, self).__init__(N_x, N_y)
        if x_min >= x_max or y_min >= y_max:
            raise Exception(f'Range incorrect: x_min = {x_min}, x_max = {x_max}, y_min = {y_min}, y_max = {y_max}')
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._family = family

    @property
    def family(self):
        return self._family

    @property
    def xlen(self):
        return self._x_max - self._x_min

    @property
    def dx(self):
        return self.xlen * self.dxAtom

    @property
    def ylen(self):
        return self._y_max - self._y_min

    @property
    def dy(self):
        return self.ylen * self.dyAtom

    @property
    def dl(self):
        return np.sqrt(self.dx**2 + self.dy**2)

    @property
    def x(self):
        return self._x_min + (self._x_max - self._x_min) * self.xAtom

    @property
    def y(self):
        return self._y_min + (self._y_max - self._y_min) * self.yAtom

    @property
    def meshgrid(self):
        return np.meshgrid(self.x, self.y)

    @property
    def xRank(self):
        return np.asarray(self._N_y * self.x.tolist())

    @property
    def yRank(self):
        yret = []
        yA = self.y
        for i in range(self._N_y):
            yret = yret + self._N_x * [yA[i]]
        return np.asarray(yret)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) > 2:
                raise Exception(f'Incorporate slice {key}')
            return self.x[key[0]], self.y[key[1]]
        return self.xRank[key], self.yRank[key]
    
    def __call__(self, func, **kwargs):
        x_list = self.xRank
        y_list = self.yRank
        fxy_list = np.zeros(self.length)
        for i in range(self.length):
            fxy_list[i] = func(x_list[i], y_list[i], **kwargs)
        return GridFunc(fxy_list, self)


class GridFunc(object):
    def __init__(self, fxy, grid):
        self._grid = grid
        self._values = np.asarray(fxy)
    
    @property
    def grid(self):
        return self._grid

    @property
    def values(self):
        return self._values

    @property
    def value(self):
        return self._values

    @property
    def value_array(self):
        return self._values.reshape(self.grid.N_y, self.grid.N_x)

    @property
    def length(self):
        return self._grid.length

    def __getitem__(self, key):
        if isinstance(key, tuple):
            raise Exception(f'Incorporate slice {key}')
        return self._grid.xRank[key], self._grid.yRank[key], self._values[key]

    def calculate_divergence_from_index(self, index):
        ix, iy = self._grid.index_rank2xy(index)
        val = self.value_array
        dx = self._grid.dx
        dy = self._grid.dy
        # df / dx
        if ix == 0:
            dfdx = (val[iy, ix + 1] - val[iy, ix]) / dx
        elif ix == self._grid.N_x - 1:
            dfdx = (val[iy, ix] - val[iy, ix-1]) / dx
        else:
            dfdx = (val[iy, ix+1] - val[iy, ix-1]) / dx / 2
        
        # df / dy
        if iy == 0:
            dfdy = (val[iy+1, ix] - val[iy, ix]) / dy
        elif iy == self._grid.N_y - 1:
            dfdy = (val[iy, ix] - val[iy-1, ix]) / dy
        else:
            dfdy = (val[iy+1, ix] - val[iy-1, ix]) / dy / 2
        return dfdx + dfdy


    def filter(self, thresh = 0.5):
        isorted = np.argsort(self._values)[::-1]
        ithresh = int(thresh * self.length)
        icollect = isorted[:ithresh]
        return GridFuncPointsCollector(self, icollect, family = self._grid.family)

    def save(self, fname):
        fpath = Path(fname)
        x = self._grid.xRank
        y = self._grid.yRank
        val = self._values
        if fpath.exists():
            with open(fpath, 'a') as f:
                for i in range(len(x)):
                    f.write("%.16e %.16e %.16e\n"%(x[i], y[i], val[i]))
        else:
            np.savetxt(fpath, np.stack([x,y,val], axis = 1))

    def save_family(self, prefix):
        prefix = Path(prefix)
        fname = prefix / f'grid_{self._grid.family}.dat'
        x = self._grid.xRank
        y = self._grid.yRank
        val = self._values
        np.savetxt(fname, np.stack([x,y,val], axis = 1))
        return


# Grid points collect
class GridFuncPointsCollector(object):
    def __init__(self, gridfunc, indexes, family = '0'):
        self._gridfunc = gridfunc
        self._indexes = indexes
        self._family = family

    @property
    def gridfunc(self):
        return self._gridfunc

    def get_points(self):
        return self._gridfunc.grid.xRank[self._indexes], self._gridfunc.grid.yRank[self._indexes], self._gridfunc.values[self._indexes]

    def __iter__(self):
        xR = self._gridfunc.grid.xRank
        yR = self._gridfunc.grid.yRank
        for ind in self._indexes:
            yield ind, xR[ind], yR[ind], self._gridfunc.values[ind]

    def iterAtom(self):
        xR = self._gridfunc.grid.xRankAtom
        yR = self._gridfunc.grid.yRankAtom
        for ind in self._indexes:
            yield ind, xR[ind], yR[ind], self._gridfunc.values[ind]

        
    # DBSCN cluster
    # Input collector of [(x,...), (y,...), (fxy,...)], and thresh epsilon
    # Return collector of indexes [(ind,...), (ind,...),...]
    def DBSCAN_cluster(self, eps = 1):
        CluIndOut = []
        eps = self._gridfunc.grid.dlAtom * eps
        eps2 = eps*eps
        unvisited = self._indexes.copy().tolist()
        visited = []
        xRA = self._gridfunc.grid.xRankAtom
        yRA = self._gridfunc.grid.yRankAtom
        while len(unvisited) > 0:
            ip = np.random.choice(unvisited)
            unvisited.remove(ip)
            visited.append(ip)
            xp = xRA[ip]
            yp = yRA[ip]
            CluNew = []
            for ind, x, y, _ in self.iterAtom():
                if np.power(x - xp,2) + np.power(y - yp,2) < eps2:
                    CluNew.append(ind)
            iiClu = 0
            while iiClu < len(CluNew):
                ipi = CluNew[iiClu]
                if ipi in unvisited:
                    unvisited.remove(ipi)
                    visited.append(ipi)
                    xi = xRA[ipi]
                    yi = yRA[ipi]
                    for ind, x, y, _ in self.iterAtom():
                        if ind not in CluNew and np.power(x - xi,2) + np.power(y - yi,2) < eps2:
                            CluNew.append(ind)
                iiClu += 1
            CluIndOut.append(CluNew)
        ret = []
        for i, idx_list in enumerate(CluIndOut):
            ret.append(GridFuncPointsCollector(self._gridfunc, np.asarray(idx_list), family = f'{self._family}_{i}'))
        return ret

    def generate_grid(self, N_x = None, N_y = None):
        xp, yp, _ = self.get_points()
        dx = self._gridfunc.grid.dx
        dy = self._gridfunc.grid.dy
        x_min = np.min(xp) - dx
        x_max = np.max(xp) + dx
        y_min = np.min(yp) - dy
        y_max = np.max(yp) + dy
        if N_x is None:
            dx_new = dx / 5.
            N_x = int( (x_max - x_min) / dx_new )
        if N_y is None:
            dy_new = dy / 5.
            N_y = int( (y_max - y_min) / dy_new )
        GrdNew = Grid2D(x_min, x_max, y_min, y_max, N_x, N_y, family = self._family)
        return GrdNew

# Used when searching for (x,y) that can maximize func(x,y)
class MultiGrid(object):
    def __init__(self, func, x_range, y_range, N_x, N_y, **kwargs):
        x_min = x_range[0]
        x_max = x_range[1]
        y_min = y_range[0]
        y_max = y_range[1]
        self._grid = Grid2D(x_min, x_max, y_min, y_max, N_x, N_y)
        self._func = func
        self._kwargs = kwargs

    @property
    def grid(self):
        return self._grid
    
    def run(self, fsave = None, eps = 1e-6, magnification = 10, filter_thresh = 0.4, maxiter = 100):
        GrdF = self._grid(self._func, **self._kwargs)
        ret = np.stack([GrdF.grid.xRank, GrdF.grid.yRank, GrdF.values], axis = 1)
        print_family = True
        if fsave is None:
            print_family = False
        if print_family:
            fpath = Path(fsave)
            if fpath.exists():
                f = open(fpath, 'w')
                f.close()
            fprefix_family = fpath.parent / 'grid_family'
            if not fprefix_family.exists():
                fprefix_family.mkdir(parents = True)
            GrdF.save(fsave)
            GrdF.save_family(fprefix_family)
        GPoints = GrdF.filter(thresh=filter_thresh)
        GPointsClusterList = GPoints.DBSCAN_cluster()
        GrdFuncList = []
        for GPCobj in GPointsClusterList:
            Grd = GPCobj.generate_grid()
            GrdF = Grd(self._func, **self._kwargs)
            ret = np.append(ret, np.stack([GrdF.grid.xRank, GrdF.grid.yRank, GrdF.values], axis = 1), axis = 0)
            if print_family:
                GrdF.save(fsave)
                GrdF.save_family(fprefix_family)
            GrdFuncList.append(GrdF)
        ind = 0
        dx_init = self._grid.dx
        dy_init = self._grid.dy
        while (len(GrdFuncList) > 0 and ind < maxiter):
            ind = ind + 1
            GrdF = GrdFuncList[0]
            # check program end
            dx = GrdF.grid.dx
            dy = GrdF.grid.dy
            div = GrdF.calculate_divergence_from_index(np.argmax(GrdF.values))
            if np.abs(div) < eps and dx < dx_init/magnification and dy < dy_init/magnification:
                GrdFuncList.remove(GrdF)
                continue
            GPoints = GrdF.filter(thresh = filter_thresh)
            GPointsClusterList = GPoints.DBSCAN_cluster()
            GrdFuncListNew = []
            for GPCobj in GrdFuncListNew:
                GrdNew = GPCobj.generate_grid()
                GrdFNew = GrdNew(self._func, **self._kwargs)
                ret = np.append(ret, np.stack([GrdFNew.grid.xRank, GrdFNew.grid.yRank, GrdFNew.values], axis = 1), axis = 0)
                if print_family:
                    GrdFNew.save(fsave)
                    GrdFNew.save_family(fprefix_family)
                GrdFuncListNew.append(GrdFNew)
            GrdFuncList.remove(GrdF)
            GrdFuncList = GrdFuncList + GrdFuncListNew
        return ret
