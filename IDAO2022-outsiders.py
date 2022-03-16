#!/usr/bin/env python
# coding: utf-8

# ## IDAO 2022 Final Solution
# #### Team: outsiders (IRAN)
# #### Team Members: 
# - Amir Abbas Bakhshipour
# - Amin Noroozi

# ### Required Libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import sys
import os
import numpy as np
import pandas as pd
from pymatgen.analysis.graphs import StructureGraph, MoleculeGraph
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yaml
import json
from pathlib import Path
from pymatgen.core import Structure
import math
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans


# ### Required Functions

# In[3]:


class coord:
    def __init__(self,x,y,z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    def distance(self, other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)
    def __add__(self, other):
        return coord(self.x+other.x,self.y+other.y,self.z+other.z)
    def __sub__(self, other):
        return coord(self.x-other.x,self.y-other.y,self.z-other.z)
    def dot(self,other):
        return (self.x*other.x+self.y*other.y+self.z*other.z)
    def absolute(self):
        return self.distance(coord(0,0,0))
    def __repr__(self):
        return "x: %s\ny: %s\nz: %s" % (self.x, self.y, self.z)


try:
    mo_x_left = -11.166104822387366

    full_mo_left = (-11.166104822387366, 21.18219065056405, 3.719751000000002)

    mo_x_right= 22.332210027612632

    full_mo_right = (22.332210027612632, 1.8419295545177048, 3.719751000000002)
    
    mo_x_range = mo_x_right - mo_x_left

    mo_y_bottom = 1.8419295545177048

    full_mo_bottom = (1.2761262879032588e-07, 1.8419295545177048, 3.719751)

    mo_y_top = 21.18219065056405

    full_mo_top = (-11.166104822387366, 21.18219065056405, 3.719751000000002)
    
    mo_y_range = mo_y_top-mo_y_bottom

    s_x_left = -9.570947227612622

    full_s_left = (-9.570947227612622, 20.261225983820975, 2.154866633304002)

    s_x_right = 23.927367622387376

    full_s_right = (23.927367622387376, 0.9209648877746301, 2.154866633304002)
    
    s_x_range = s_x_right - s_x_left

    s_y_bottom = 0.9209648877746301

    full_s_bottom = (1.5951577223873723, 0.9209648877746301, 2.1548666333040005)

    s_y_top = 20.261225983820975

    full_s_top = (-9.570947227612622, 20.261225983820975, 2.154866633304002)
    
    s_y_range = s_y_top - s_y_bottom
except:
    pass

x_unit = coord(3.1903158276126287-1.2761262879032588e-07,0,0)
y_unit = coord(-1.5951577223873705, 4.60482399681004, 3.719751)-coord(1.2761262879032588e-07, 1.8419295545177048, 3.719751)
mo_y_dir = coord(-11.166104822387366, 21.18219065056405, 3.719751)-coord(1.2761262879032588e-07, 1.8419295545177048, 3.719751)
s_y_dir = coord(-9.570947227612622, 20.261225983820975, 2.154866633304)-coord(1.5951577223873723, 0.9209648877746301, 2.154866633304)


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)

def prepare_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }

    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values(), targets=targets)
    # train_test_split(data, test_size=0.25, random_state=666)
    return data

def test_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }
    data = pd.DataFrame(columns=["id","structures"])
    data = data.assign(id=struct.keys())
    data = data.assign(structures=struct.values())

    return data


def distance_3d(u, v):
    """
        this function computes euclidean distance for 3d vectores
    """
    return math.sqrt((u[0]-v[0])**2 + (u[1]-v[1])**2 + (u[2]-v[2])**2)

def atom_counter(structure):
    """
      this function counts the atoms (Mo, S, W, Se) and
      the number of defects and vacancies in each structure
    """
    mo = 0
    s = 0
    w = 0
    se = 0
    for i in range(len(structure)):
        a = structure[i].species.alphabetical_formula
        if a == 'Mo1':
            mo += 1
        elif a == 'S1':
            s += 1
        elif a == 'W1':
            w += 1
        elif a == 'Se1':
            se += 1
    return {
        'mo': mo,
        's': s,
        'w': w,
        'se': se,
        'defects': se+w,
        'vacancies':192-(mo+s+w+se)
    }

def train_base_features(st_df):
    my_df = []
    for i in range(len(st_df)):
        structure = st_df['structure'].iloc[i]
        d = {}
        d['id'] = st_df['id'].iloc[i]
        d['target'] = st_df['target'].iloc[i]
        atom_counts = atom_counter(structure)
        d['def_num'] = atom_counts['defects']
        defect_coords = defect_coords_finder(structure)
        defect_orients = defect_orient_finder(structure)
        for i,defect_coord in enumerate(defect_coords):
            if defect_coord[0] == 'Se1':
                d['def%s_type'%(i+1)] = 1
            elif defect_coord[0] == 'W1':
                d['def%s_type'%(i+1)] = 2
            d['def%s_ori'%(i+1)] = defect_coord[1]
            d['def%s_x'%(i+1)] = defect_coord[2]
            d['def%s_y'%(i+1)] = defect_coord[3]
            d['def%s_z'%(i+1)] = defect_coord[4]
        d['vac_num'] = atom_counts['vacancies']
        vacancy_coords = vacancy_coords_finder(structure)
        top = 0
        bottom = 0
        middle = 0
        for i,vacancy_coord in enumerate(vacancy_coords):
            if vacancy_type(vacancy_coord) == 'Mo':
                d['vac%s_type'%(i+1)] = 0
                middle += 1
            elif vacancy_type(vacancy_coord) == 'Sd':
                d['vac%s_type'%(i+1)] = -1
                bottom += 1
            elif vacancy_type(vacancy_coord) == 'Su':
                d['vac%s_type'%(i+1)] = 1
                top += 1
            d['vac%s_x'%(i+1)] = vacancy_coord[0]
            d['vac%s_y'%(i+1)] = vacancy_coord[1]
            d['vac%s_z'%(i+1)] = vacancy_coord[2]
        d['num_bottom_vac'] = bottom
        d['num_middle_vac'] = middle
        d['num_top_vac'] = top
        my_df.append(d)
    df = pd.DataFrame(my_df) 
    cols = ['id',
            'def_num',
            'def1_type', 'def1_ori', 'def1_x', 'def1_y', 'def1_z',
            'def2_type', 'def2_ori', 'def2_x', 'def2_y', 'def2_z',
            'def3_type', 'def3_ori', 'def3_x', 'def3_y', 'def3_z',
            'vac_num', 'num_bottom_vac', 'num_middle_vac', 'num_top_vac',
            'vac1_type', 'vac1_x', 'vac1_y', 'vac1_z',
            'vac2_type', 'vac2_x', 'vac2_y', 'vac2_z',
            'vac3_type', 'vac3_x', 'vac3_y', 'vac3_z',
            'target'
           ]
    train_df = pd.DataFrame(columns=cols)
    train_df['id']=df['id']
    train_df['def_num']=df['def_num']
    try:
        train_df['def1_type']=df['def1_type']
    except:
        pass
    try:
        train_df['def1_ori']=df['def1_ori']
    except:
        pass
    try:
        train_df['def1_x']=df['def1_x']
    except:
        pass
    try:
        train_df['def1_y']=df['def1_y']
    except:
        pass
    try:
        train_df['def1_z']=df['def1_z']
    except:
        pass


    try:
        train_df['def2_type']=df['def2_type']
    except:
        pass
    try:
        train_df['def2_ori']=df['def2_ori']
    except:
        pass
    try:  
        train_df['def2_x']=df['def2_x']
    except:
        pass
    try:  
        train_df['def2_y']=df['def2_y']
    except:
        pass
    try:  
        train_df['def2_z']=df['def2_z']
    except:
        pass

  
    try:  
        train_df['def3_type']=df['def3_type']
    except:
        pass
    try:
        train_df['def3_ori']=df['def3_ori']
    except:
        pass
    try:   
        train_df['def3_x']=df['def3_x']
    except:
        pass
    try:  
        train_df['def3_y']=df['def3_y']
    except:
        pass
    try:  
        train_df['def3_z']=df['def3_z']
    except:
        pass 


    try:  
        train_df['vac_num']=df['vac_num']
    except:
        pass
    try:  
        train_df['num_bottom_vac']=df['num_bottom_vac']
    except:
        pass
    try:  
        train_df['num_middle_vac']=df['num_middle_vac']
    except:
        pass
    try:  
        train_df['num_top_vac']=df['num_top_vac']
    except:
        pass

    try:  
        train_df['vac1_type']=df['vac1_type']
    except:
        pass
    try:  
        train_df['vac1_x']=df['vac1_x']
    except:
        pass
    try:  
        train_df['vac1_y']=df['vac1_y']
    except:
        pass
    try:  
        train_df['vac1_z']=df['vac1_z']
    except:
        pass
  
    try:  
        train_df['vac2_type']=df['vac2_type']
    except:
        pass
    try:  
        train_df['vac2_x']=df['vac2_x']
    except:
        pass
    try:  
        train_df['vac2_y']=df['vac2_y']
    except:
        pass
    try:  
        train_df['vac2_z']=df['vac2_z']
    except:
        pass

    try:  
        train_df['vac3_type']=df['vac3_type']
    except:
        pass
    try:  
        train_df['vac3_x']=df['vac3_x']
    except:
        pass
    try:  
        train_df['vac3_y']=df['vac3_y']
    except:
        pass
    try:  
        train_df['vac3_z']=df['vac3_z']
    except:
        pass
    train_df['target']=df['target']
    train_df.fillna(0, inplace=True)
    return train_df

def test_base_features(st_df):
    my_df = []
    for i in range(len(st_df)):
        structure = st_df['structures'].iloc[i]
        d = {}
        d['id'] = st_df['id'].iloc[i]
        atom_counts = atom_counter(structure)
        d['def_num'] = atom_counts['defects']
        defect_coords = defect_coords_finder(structure)
        defect_orients = defect_orient_finder(structure)
        for i,defect_coord in enumerate(defect_coords):
            if defect_coord[0] == 'Se1':
                d['def%s_type'%(i+1)] = 1
            elif defect_coord[0] == 'W1':
                d['def%s_type'%(i+1)] = 2
            d['def%s_ori'%(i+1)] = defect_coord[1]
            d['def%s_x'%(i+1)] = defect_coord[2]
            d['def%s_y'%(i+1)] = defect_coord[3]
            d['def%s_z'%(i+1)] = defect_coord[4]
        d['vac_num'] = atom_counts['vacancies']
        vacancy_coords = vacancy_coords_finder(structure)
        top = 0
        bottom = 0
        middle = 0
        
        for i,vacancy_coord in enumerate(vacancy_coords):
            if vacancy_type(vacancy_coord) == 'Mo':
                d['vac%s_type'%(i+1)] = 0
                middle += 1
            elif vacancy_type(vacancy_coord) == 'Sd':
                d['vac%s_type'%(i+1)] = -1
                bottom += 1
            elif vacancy_type(vacancy_coord) == 'Su':
                d['vac%s_type'%(i+1)] = 1
                top += 1
            d['vac%s_x'%(i+1)] = vacancy_coord[0]
            d['vac%s_y'%(i+1)] = vacancy_coord[1]
            d['vac%s_z'%(i+1)] = vacancy_coord[2]
        d['num_bottom_vac'] = bottom
        d['num_middle_vac'] = middle
        d['num_top_vac'] = top
        my_df.append(d)
    df = pd.DataFrame(my_df) 
    cols = ['id',
            'def_num',
            'def1_type',
            'def1_ori',
            'def1_x', 'def1_y', 'def1_z',
            'def2_type',
            'def2_ori',
            'def2_x', 'def2_y', 'def2_z',
            'def3_type',
            'def3_ori',
            'def3_x', 'def3_y', 'def3_z',
            'vac_num',
            'num_bottom_vac', 'num_middle_vac', 'num_top_vac',
            'vac1_type', 'vac1_x', 'vac1_y', 'vac1_z',
            'vac2_type', 'vac2_x', 'vac2_y', 'vac2_z',
            'vac3_type', 'vac3_x', 'vac3_y', 'vac3_z',
           ]
    train_df = pd.DataFrame(columns=cols)
    train_df['id']=df['id']
    train_df['def_num']=df['def_num']
    try:
        train_df['def1_type']=df['def1_type']
    except:
        pass
    try:
        train_df['def1_ori']=df['def1_ori']
    except:
        pass
    try:
        train_df['def1_x']=df['def1_x']
    except:
        pass
    try:
        train_df['def1_y']=df['def1_y']
    except:
        pass
    try:
        train_df['def1_z']=df['def1_z']
    except:
        pass


    try:
        train_df['def2_type']=df['def2_type']
    except:
        pass
    try:
        train_df['def2_ori']=df['def2_ori']
    except:
        pass
    try:  
        train_df['def2_x']=df['def2_x']
    except:
        pass
    try:  
        train_df['def2_y']=df['def2_y']
    except:
        pass
    try:  
        train_df['def2_z']=df['def2_z']
    except:
        pass

  
    try:  
        train_df['def3_type']=df['def3_type']
    except:
        pass
    try:
        train_df['def3_ori']=df['def3_ori']
    except:
        pass
    try:   
        train_df['def3_x']=df['def3_x']
    except:
        pass
    try:  
        train_df['def3_y']=df['def3_y']
    except:
        pass
    try:  
        train_df['def3_z']=df['def3_z']
    except:
        pass 


    try:  
        train_df['vac_num']=df['vac_num']
    except:
        pass
    try:  
        train_df['num_bottom_vac']=df['num_bottom_vac']
    except:
        pass
    try:  
        train_df['num_middle_vac']=df['num_middle_vac']
    except:
        pass
    try:  
        train_df['num_top_vac']=df['num_top_vac']
    except:
        pass

    try:  
        train_df['vac1_type']=df['vac1_type']
    except:
        pass
    try:  
        train_df['vac1_x']=df['vac1_x']
    except:
        pass
    try:  
        train_df['vac1_y']=df['vac1_y']
    except:
        pass
    try:  
        train_df['vac1_z']=df['vac1_z']
    except:
        pass
  
    try:  
        train_df['vac2_type']=df['vac2_type']
    except:
        pass
    try:  
        train_df['vac2_x']=df['vac2_x']
    except:
        pass
    try:  
        train_df['vac2_y']=df['vac2_y']
    except:
        pass
    try:  
        train_df['vac2_z']=df['vac2_z']
    except:
        pass

    try:  
        train_df['vac3_type']=df['vac3_type']
    except:
        pass
    try:  
        train_df['vac3_x']=df['vac3_x']
    except:
        pass
    try:  
        train_df['vac3_y']=df['vac3_y']
    except:
        pass
    try:  
        train_df['vac3_z']=df['vac3_z']
    except:
        pass
    train_df.fillna(0, inplace=True)
    return train_df

def defect_coords_finder(structure):
    """
        this function finds the 3d coordinates of each defect in structure
    """
    defect_coords = []
    atom_count = atom_counter(structure)
    defects = atom_count['defects']
    first_defect_idx = atom_count['mo']
    for defect in range(defects):
        name = structure[first_defect_idx+defect].species.alphabetical_formula
        x = structure[first_defect_idx+defect].x
        y = structure[first_defect_idx+defect].y
        z = structure[first_defect_idx+defect].z
        def_ori = 0
        for coord in mo_coords:
                if coord[0] == x and coord[1] == y:
                    if z > coord[2]:
                        def_ori = 1
                    elif z < coord[2]:
                        def_ori = -1
        for coord in s_coords:
                if coord[0] == x and coord[1] == y:
                    if z > coord[2]:
                        def_ori = 1
                    elif z < coord[2]:
                        def_ori = -1
        defect_coords.append((name,def_ori,x,y,z))
    return defect_coords

def defect_orient_finder(structure):
    """
        this function finds the (a,b,c) of each defect in structure
    """
    defect_orient = []
    atom_count = atom_counter(structure)
    defects = atom_count['defects']
    first_defect_idx = atom_count['mo']
    for defect in range(defects):
        name = structure[first_defect_idx+defect].species.alphabetical_formula
        a = structure[first_defect_idx+defect].a
        b = structure[first_defect_idx+defect].b
        c = structure[first_defect_idx+defect].c
        defect_orient.append((name,a,b,c))
    return defect_orient

def vacancy_coords_finder(structure):
    """
        this function finds the coordinates of each vacancy in structure
    """
    err = 1e-3
    vacancy_coords = []
    atom_count = atom_counter(structure)
    total_atoms = atom_count['mo'] + atom_count['s'] + atom_count['defects']
    vacancies = atom_count['vacancies']
    mo = atom_count['mo']
    s = atom_count['s']
    mo_coords_copy = mo_coords.copy()
    for atom in range(192-s):
        tmp = structure[atom]
        coords = tuple(tmp.coords)
        for i,u in enumerate(mo_coords_copy):
            if distance_3d(u,coords) < err:
                mo_coords_copy.pop(i)
    if len(mo_coords_copy) != 0:
        for idx in range(len(mo_coords_copy)):
            x = mo_coords_copy[idx][0]
            y = mo_coords_copy[idx][1]
            z = mo_coords_copy[idx][2]
            vacancy_coords.append((x,y,z))
    s_coords_copy = s_coords.copy()
    for atom in range(total_atoms-mo):
        tmp = structure[atom+mo]
        coords = tuple(tmp.coords)
        for i,u in enumerate(s_coords_copy):
            if distance_3d(coords,u) < err:
                s_coords_copy.pop(i)
  
    if len(s_coords_copy) != 0:
        for idx in range(len(s_coords_copy)):
            x = s_coords_copy[idx][0]
            y = s_coords_copy[idx][1]
            z = s_coords_copy[idx][2]
            vacancy_coords.append((x,y,z))
    return vacancy_coords

def vacancy_type(vacancy_coords):
    if vacancy_coords in mo_coords:
        return 'Mo'
    elif abs(vacancy_coords[2]-2.1) < 1:
        return 'Sd'
    return 'Su'

def defect_locator(structure):
    """
      this function finds the location of each defect in 8x8 matrix
    """
    bottom = 2.154867
    middle = 3.719751
    top = 5.284635
    err = 1e-5
    defects = defect_coords_finder(structure)
    defect_locations = []
    for defect in defects:
        if abs(defect[3]-top) < err:
            for yidx,y in enumerate(s_ys):
                if abs(y-defect[2]) < err:
                    defy = y
                    ypixel = yidx
                    break
            for xidx,x in enumerate(s_xs[str(y)]):
                if abs(x-defect[1]) < err:
                    defx = x
                    xpixel = xidx
                    break
            defect_locations.append((xpixel,ypixel))
        elif abs(defect[3]-bottom) < err:
            for yidx,y in enumerate(s_ys):
                if abs(y-defect[2]) < err:
                    defy = y
                    ypixel = yidx
                    break
            for xidx,x in enumerate(s_xs[str(y)]):
                if abs(x-defect[1]) < err:
                    defx = x
                    xpixel = xidx
                    break
            defect_locations.append((xpixel,ypixel))
        elif abs(defect[3]-middle) < err:
            for yidx,y in enumerate(mo_ys):
                if abs(y-defect[2]) < err:
                    defy = y
                    ypixel = yidx
                    break
            for xidx,x in enumerate(mo_xs[str(y)]):
                if abs(x-defect[1]) < err:
                    defx = x
                    xpixel = xidx
                    break
            defect_locations.append((xpixel,ypixel))
    return defect_locations    

def vacancy_locator(structure):
    """
      this function finds the location of each vacancy in 8x8 matrix
    """
    bottom = 2.154867
    middle = 3.719751
    top = 5.284635
    err = 1e-5
    vacancies = vacancy_coords_finder(structure)
    vacancy_locations = []
    for vacancy in vacancies:
        if abs(vacancy[2]-top) < err:
            for yidx,y in enumerate(s_ys):
                if abs(y-vacancy[1]) < err:
                    vacy = y
                    ypixel = yidx
                    break
            for xidx,x in enumerate(s_xs[str(y)]):
                if abs(x-vacancy[0]) < err:
                    vacx = x
                    xpixel = xidx
                    break
            vacancy_locations.append((xpixel,ypixel))
        elif abs(vacancy[2]-bottom) < err:
            for yidx,y in enumerate(s_ys):
                if abs(y-vacancy[1]) < err:
                    vacy = y
                    ypixel = yidx
                    break
            for xidx,x in enumerate(s_xs[str(y)]):
                if abs(x-vacancy[0]) < err:
                    vacx = x
                    xpixel = xidx
                    break
            vacancy_locations.append((xpixel,ypixel))
        elif abs(vacancy[2]-middle) < err:
            for yidx,y in enumerate(mo_ys):
                if abs(y-vacancy[1]) < err:
                    vacy = y
                    ypixel = yidx
                    break
            for xidx,x in enumerate(mo_xs[str(y)]):
                if abs(x-vacancy[0]) < err:
                    vacx = x
                    xpixel = xidx
                    break
            vacancy_locations.append((xpixel,ypixel))
    return vacancy_locations

def influence(matrix_size, position, value, falloff):
    """
      this function computes the influence propagation
    """
    matrix = [[None for i in range(matrix_size)] for j in range(matrix_size)]
    x, y = position
    matrix[x][y] = value
    q = []
    q.append((x,y))
    while(len(q) != 0):
        x, y = q.pop(0)
        for i in range(matrix_size):
            if matrix[x][i] is None:
                matrix[x][i] = matrix[x][y] - abs(i-y) * falloff
                q.append((x,i))
            if matrix[i][y] is None:
                matrix[i][y] = matrix[x][y] - abs(i-x) * falloff
                q.append((i,y))
    return matrix

def feature_extractor(structure):
    """
        this function gives a 64x1 (flattened 8x8) influce propagated feature map
        for each structure
    """
    defect_pixels = defect_locator(structure)
    vacancy_pixels = vacancy_locator(structure)
    feature_map = np.zeros((64,1), dtype=np.float32)
    for defect in defect_pixels:
        matrix = influence(8, (defect[0],defect[1]), 10, 1)
        flattened = np.zeros((64,1), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                flattened[8*i+j] = matrix[i][j]
        feature_map = feature_map + flattened
    for vacancy in vacancy_pixels:
        matrix = influence(8, (vacancy[0],vacancy[1]), 10, 1)
        flattened = np.zeros((64,1), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                flattened[8*i+j] = matrix[i][j]
        feature_map = feature_map + flattened
    return feature_map

def cluster_target(df, n_clusters):
    """
        this function split a dataframe to multiple dataframes base on
        target distribution
    """
    target = np.copy(np.asarray(df['target']).reshape((-1,1)))
    model = KMeans(n_clusters=n_clusters, random_state=13740103)
    model.fit(target)
    new_df = df.copy()
    new_df['class'] = model.labels_
    means = list(model.cluster_centers_)
    return new_df, means

def class_based_dataframe_split(df,n_classes):
    dataframes = []
    for i in range(n_classes):
        new_df = df.loc[df['class'] == i]
        for x in ['target']:
            q75,q25 = np.percentile(new_df.loc[:,x],[75,25])
            intr_qr = q75-q25
            mx = q75+(intr_qr)
            mn = q25-(intr_qr)
            new_df.loc[new_df[x] < mn,x] = np.nan
            new_df.loc[new_df[x] > mx,x] = np.nan
            new_df.dropna(axis = 0, inplace=True)
        dataframes.append(new_df)
    return dataframes

def class_based_dataframe_split_test(df,n_classes):
    dataframes = []
    for i in range(n_classes):
        new_df = df.loc[df['class'] == i]
        dataframes.append(new_df)
    return dataframes

def generate_standard_dataframe(df):
    my_df = []
    for i in range(len(df)):
        d = {}
        d['id'] = df.index[i]
        d['structure'] = df['structures'].iloc[i]
        d['target'] = df['targets'].iloc[i]
        my_df.append(d)
    return pd.DataFrame(my_df)

def z_identifier(z):
    if abs(z-3.71) < 0.1:
        return 'middle'
    if abs(z-2.15) < 0.1:
        return 'bottom'
    return 'top'

def ewa_score(preds, targets):
    cnt = 0
    n = len(preds)
    for i in range(n):
        if abs(preds[i]-targets[i]) < 0.02:
            cnt += 1
    return cnt/n

def atom_count(df):
    bigl = []
    for i in list(df.index):
        mo = 0
        s = 0
        w = 0
        se = 0
        for j in range(len(df['structures'][i])):
            a = df['structures'][i][j].species.alphabetical_formula
            if a == 'Mo1':
                mo += 1
            elif a == 'S1':
                s += 1
            elif a == 'W1':
                w += 1
            elif a == 'Se1':
                se += 1
        bigl.append((i,mo,s,w,se))
    return bigl

def find_atom_coords(df, bigl):
    for i in bigl:
        if i[1] == 64:
            found = i[0]
            break
  
    mo_coords = []
    for i in range(64):
        tmp = df['structures'][found][i]
        mo_coords.append((tuple(tmp.coords)))

    for i in bigl:
        if i[2] == 128:
            found = i[0]
            mos = i[1]+i[4]+i[3]
            break

    s_coords = []
    for i in range(128):
        tmp = df['structures'][found][i+mos]
        s_coords.append((tuple(tmp.coords)))
  
    return mo_coords, s_coords

def find_atom_xys(mo_coords, s_coords):
    mo_xy = np.zeros((len(mo_coords),2), dtype=np.float64)
    s_xy = np.zeros((int(len(s_coords)/2),2), dtype=np.float64)
    for i,st in enumerate(mo_coords):
        mo_xy[i,0] = st[0]
        mo_xy[i,1] = st[1]

    for i,st in enumerate(s_coords[:64]):
        s_xy[i,0] = st[0]
        s_xy[i,1] = st[1]
  
    mo_ys = []
    for i in range(mo_xy.shape[0]):
        if mo_xy[i,1] not in mo_ys:
            mo_ys.append(mo_xy[i,1])

    s_ys = []
    for i in range(s_xy.shape[0]):
        if s_xy[i,1] not in s_ys:
            s_ys.append(s_xy[i,1])

    mo_xs = {
    '1.8419295545177048':[],
    '4.60482399681004':[],
    '7.367718512779559':[],
    '10.13061288139471':[],
    '12.893507323687045':[],
    '15.656401765979378':[],
    '18.419296208271714':[],
    '21.18219065056405':[]
  }
    s_xs = {
    '0.9209648877746301':[],
    '3.683859330066965':[],
    '6.4467537723593':[],
    '9.209648214651635':[],
    '11.97254265694397':[],
    '14.735437025559118':[],
    '17.49833154152864':[],
    '20.261225983820975':[]
  }

    err = 1e-2
    for i in range(mo_xy.shape[0]):
        for j in mo_ys:
            if abs(mo_xy[i,1] - j) < err:
                mo_xs[str(j)].append(mo_xy[i,0])

    for i in range(s_xy.shape[0]):
        for j in s_ys:
            if abs(s_xy[i,1]-j) < err:
                s_xs[str(j)].append(s_xy[i,0])

    return mo_xs, mo_ys, s_xs, s_ys

def generate_dataframe(df, def_num, vac_num):
    new_df = pd.DataFrame()
    for i in range(len(df)):
        if df.loc[i,'def_num'] == def_num and df.loc[i,'vac_num'] == vac_num:
            new_df = new_df.append(df.loc[i,:])
    for i in range(def_num+1, 4):
        try:
            new_df.drop(['def%s_type'%i,'def%s_ori'%i,'def%s_x'%i,'def%s_y'%i,'def%s_z'%i], axis=1, inplace=True)
        except:
            pass
    for i in range(vac_num+1, 4):
        try:
            new_df.drop(['vac%s_type'%i,'vac%s_x'%i,'vac%s_y'%i,'vac%s_z'%i],axis=1,inplace=True)
        except:
            pass
    if def_num == 0 or vac_num == 0:
        try:
            new_df.drop(['def_num', 'vac_num'], axis=1,inplace=True)
        except:
            pass
    return new_df


# ### Data Loading

# In[4]:


train = prepare_dataset('./data/dichalcogenides_public')
targets = pd.read_csv('./data/dichalcogenides_public/targets.csv')
pv_test = test_dataset('./data/dichalcogenides_private/')


# ### Data Preprocessing

# #### Extracting base features for training data

# In[5]:


bigl = atom_count(train)
mo_coords, s_coords = find_atom_coords(train, bigl)
mo_xs, mo_ys, s_xs, s_ys = find_atom_xys(mo_coords, s_coords)
bf = train_base_features(generate_standard_dataframe(train))


# #### Dataframe split by number of defects and vacancies

# In[125]:


all_train_dfs = {}
for defect in range(0,4):
    for vacancy in range(0,4):
        new_df = generate_dataframe(bf, defect, vacancy)
        if len(new_df) != 0:
            all_train_dfs['D%sV%s'%(str(defect),str(vacancy))] = new_df


# #### Prepare features for D0V2 train

# In[126]:


#best solution
d0v2_df = all_train_dfs['D0V2']
one_two = []
for i in range(len(d0v2_df)):
    x1 = d0v2_df['vac1_x'].iloc[i]
    y1 = d0v2_df['vac1_y'].iloc[i]
    z1 = d0v2_df['vac1_z'].iloc[i]
    x2 = d0v2_df['vac2_x'].iloc[i]
    y2 = d0v2_df['vac2_y'].iloc[i]
    z2 = d0v2_df['vac2_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))
new_df = pd.DataFrame(columns=['id', 'vac1_type', 'vac2_type', 'one_two', 'target'])
new_df['id'] = d0v2_df['id']
new_df['vac1_type'] = d0v2_df['vac1_type']
new_df['vac2_type'] = d0v2_df['vac2_type']
new_df['one_two'] = one_two
new_df['target'] = d0v2_df['target']
d0v2_df = new_df


# #### Prepare features for D0V3

# In[127]:


#new
d0v3_df = all_train_dfs['D0V3']
one_two = []
one_three = []
two_three = []
for i in range(len(d0v3_df)):
    x1 = d0v3_df['vac1_x'].iloc[i]
    y1 = d0v3_df['vac1_y'].iloc[i]
    z1 = d0v3_df['vac1_z'].iloc[i]
    x2 = d0v3_df['vac2_x'].iloc[i]
    y2 = d0v3_df['vac2_y'].iloc[i]
    z2 = d0v3_df['vac2_z'].iloc[i]
    x3 = d0v3_df['vac3_x'].iloc[i]
    y3 = d0v3_df['vac3_y'].iloc[i]
    z3 = d0v3_df['vac3_z'].iloc[i]

    vac_coords = [coord(x1,y1,z1),coord(x2,y2,z2),coord(x3,y3,z3)]
    
    diff_vectors = [
        coord(mo_x_range,0,0)+x_unit,
        coord(-mo_x_range,0,0)-x_unit,
        mo_y_dir+y_unit,
        coord(0,0,0)-mo_y_dir-y_unit,
        coord(mo_x_range,0,0)+x_unit+mo_y_dir+y_unit,
        coord(mo_x_range,0,0)+x_unit-mo_y_dir-y_unit,
        coord(-mo_x_range,0,0)-x_unit+mo_y_dir+y_unit,
        coord(-mo_x_range,0,0)-x_unit-mo_y_dir-y_unit,    
    ]
    
    all_one = [vac_coords[0]]
    all_two = [vac_coords[1]]
    all_three = [vac_coords[2]]
    for diff in diff_vectors:
        all_one.append(vac_coords[0]+diff)
        all_two.append(vac_coords[1]+diff)
        all_three.append(vac_coords[2]+diff)
    min_one_two = 1e5
    min_one_three = 1e5
    min_two_three = 1e5
    for pos in all_two:
        tmp1 = vac_coords[0].distance(pos)
        tmp2 = vac_coords[2].distance(pos)
        if tmp1 < min_one_two:
            min_one_two = tmp1
        if tmp2 < min_two_three:
            min_two_three = tmp2
    for pos in all_three:
        tmp = vac_coords[0].distance(pos)
        if tmp < min_one_three:
            min_one_three = tmp
    
    one_two.append(min_one_two)
    one_three.append(min_one_three)
    two_three.append(min_two_three)    

new_df = pd.DataFrame(
    columns=['id', 'one_two', 'one_three', 'two_three','target'])
new_df['id'] = d0v3_df['id']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
new_df['target'] = d0v3_df['target']
d0v3_df = new_df


# #### Prepare features for D1V1

# In[128]:


#best solution
d1v1_df = all_train_dfs['D1V1']
one_two = []
for i in range(len(d1v1_df)):
    x1 = d1v1_df['def1_x'].iloc[i]
    y1 = d1v1_df['def1_y'].iloc[i]
    z1 = d1v1_df['def1_z'].iloc[i]
    x2 = d1v1_df['vac1_x'].iloc[i]
    y2 = d1v1_df['vac1_y'].iloc[i]
    z2 = d1v1_df['vac1_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))
new_df = pd.DataFrame(columns=['id', 'vac1_type', 'def1_type', 'def1_ori', 'one_two', 'target'])
new_df['id'] = d1v1_df['id']
new_df['vac1_type'] = d1v1_df['vac1_type']
new_df['def1_type'] = d1v1_df['def1_type']
new_df['def1_ori'] = d1v1_df['def1_ori']
new_df['one_two'] = one_two
new_df['target'] = d1v1_df['target']
d1v1_df = new_df


# #### Preparing features for D1V2

# In[129]:


#best solution
d1v2_df = all_train_dfs['D1V2']
one_two = []
one_three = []
two_three = []
for i in range(len(d1v2_df)):
    x1 = d1v2_df['vac1_x'].iloc[i]
    y1 = d1v2_df['vac1_y'].iloc[i]
    z1 = d1v2_df['vac1_z'].iloc[i]
    x2 = d1v2_df['vac2_x'].iloc[i]
    y2 = d1v2_df['vac2_y'].iloc[i]
    z2 = d1v2_df['vac2_z'].iloc[i]
    x3 = d1v2_df['def1_x'].iloc[i]
    y3 = d1v2_df['def1_y'].iloc[i]
    z3 = d1v2_df['def1_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))
    one_three.append(distance_3d((x1, y1, z1), (x3, y3, z3)))
    two_three.append(distance_3d((x2, y2, z2), (x3, y3, z3)))

new_df = pd.DataFrame(
    columns=['id', 'num_bottom_vac', 'num_middle_vac', 'num_top_vac', 'vac1_type', 'vac2_type', 'def1_type', 'def1_ori',
             'one_two', 'one_three', 'two_three', 'target'])
new_df['id'] = d1v2_df['id']
new_df['num_bottom_vac'] = d1v2_df['num_bottom_vac']
new_df['num_middle_vac'] = d1v2_df['num_middle_vac']
new_df['num_top_vac'] = d1v2_df['num_top_vac']
new_df['vac1_type'] = d1v2_df['vac1_type']
new_df['vac2_type'] = d1v2_df['vac2_type']
new_df['def1_type'] = d1v2_df['def1_type']
new_df['def1_ori'] = d1v2_df['def1_ori']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
new_df['target'] = d1v2_df['target']
d1v2_df = new_df


# #### Preparing features for D2V0

# In[130]:


#best solution
d2v0_df = all_train_dfs['D2V0']
one_two = []
for i in range(len(d2v0_df)):
    x1 = d2v0_df['def1_x'].iloc[i]
    y1 = d2v0_df['def1_y'].iloc[i]
    z1 = d2v0_df['def1_z'].iloc[i]
    x2 = d2v0_df['def2_x'].iloc[i]
    y2 = d2v0_df['def2_y'].iloc[i]
    z2 = d2v0_df['def2_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))

new_df = pd.DataFrame(columns=['id', 'def1_type', 'def1_ori', 'def2_type', 'def2_ori', 'one_two', 'target'])
new_df['id'] = d2v0_df['id']
new_df['def1_type'] = d2v0_df['def1_type']
new_df['def1_ori'] = d2v0_df['def1_ori']
new_df['def2_type'] = d2v0_df['def2_type']
new_df['def2_ori'] = d2v0_df['def2_ori']
new_df['one_two'] = one_two
new_df['target'] = d2v0_df['target']
d2v0_df = new_df


# #### Preparing features for D2V1

# In[131]:


#best solution
d2v1_df = all_train_dfs['D2V1']
one_two = []
one_three = []
two_three = []
for i in range(len(d2v1_df)):
    x1 = d2v1_df['vac1_x'].iloc[i]
    y1 = d2v1_df['vac1_y'].iloc[i]
    z1 = d2v1_df['vac1_z'].iloc[i]
    x2 = d2v1_df['def1_x'].iloc[i]
    y2 = d2v1_df['def1_y'].iloc[i]
    z2 = d2v1_df['def1_z'].iloc[i]
    x3 = d2v1_df['def2_x'].iloc[i]
    y3 = d2v1_df['def2_y'].iloc[i]
    z3 = d2v1_df['def2_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))
    one_three.append(distance_3d((x1, y1, z1), (x3, y3, z3)))
    two_three.append(distance_3d((x2, y2, z2), (x3, y3, z3)))

new_df = pd.DataFrame(
    columns=['id', 'vac1_type', 'def1_type', 'def1_ori', 'def2_type', 'def2_ori', 'one_two', 'one_three', 'two_three',
             'target'])
new_df['id'] = d2v1_df['id']
new_df['vac1_type'] = d2v1_df['vac1_type']
new_df['def1_type'] = d2v1_df['def1_type']
new_df['def1_ori'] = d2v1_df['def1_ori']
new_df['def2_type'] = d2v1_df['def2_type']
new_df['def2_ori'] = d2v1_df['def2_ori']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
new_df['target'] = d2v1_df['target']
d2v1_df = new_df


# #### Preparing features for D3V0

# In[132]:


#new
d3v0_df = all_train_dfs['D3V0']
one_two = []
one_three = []
two_three = []
for i in range(len(d3v0_df)):
    x1 = d3v0_df['def1_x'].iloc[i]
    y1 = d3v0_df['def1_y'].iloc[i]
    z1 = d3v0_df['def1_z'].iloc[i]
    x2 = d3v0_df['def2_x'].iloc[i]
    y2 = d3v0_df['def2_y'].iloc[i]
    z2 = d3v0_df['def2_z'].iloc[i]
    x3 = d3v0_df['def3_x'].iloc[i]
    y3 = d3v0_df['def3_y'].iloc[i]
    z3 = d3v0_df['def3_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))
    one_three.append(distance_3d((x1, y1, z1), (x3, y3, z3)))
    two_three.append(distance_3d((x2, y2, z2), (x3, y3, z3)))

new_df = pd.DataFrame(
    columns=['id', 'def1_type', 'def1_ori', 'def2_type', 'def2_ori', 'def3_type', 'def3_ori', 'one_two', 'one_three',
             'two_three', 'target'])
new_df['id'] = d3v0_df['id']
new_df['def1_type'] = d3v0_df['def1_type']
new_df['def1_ori'] = d3v0_df['def1_ori']
new_df['def2_type'] = d3v0_df['def2_type']
new_df['def2_ori'] = d3v0_df['def2_ori']
new_df['def3_type'] = d3v0_df['def3_type']
new_df['def3_ori'] = d3v0_df['def3_ori']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
new_df['target'] = d3v0_df['target']
d3v0_df = new_df


# ### Making Decision Tree Regressor model for each split

# #### D0V1 (1 sample)

# In[133]:


d0v1_output = np.float64(all_train_dfs['D0V1']['target'])


# #### D0V2 (20 samples)

# In[134]:


first = d0v2_df
d0v2_xtrain = first.iloc[:, 1:len(first.columns) - 1].values
d0v2_ytrain = first.loc[:, 'target'].values
md = 2
d0v2 = DecisionTreeRegressor(max_depth=md, criterion='absolute_error', random_state=13740103)
d0v2.fit(d0v2_xtrain, d0v2_ytrain)


# #### D0V3 (363 samples)

# In[135]:


first, second = train_test_split(d0v3_df, test_size=0.2, random_state=13740103)
d0v3_xtrain = first.iloc[:, 1:len(first.columns) - 1].values
d0v3_ytrain = first.loc[:,'target'].values
d0v3_xtest = second.iloc[:,1:len(second.columns)-1].values
d0v3_ytest = second.loc[:,'target'].values
md = 10
d0v3 = DecisionTreeRegressor(max_depth=md, criterion='absolute_error', random_state=13740103)
d0v3.fit(d0v3_xtrain, d0v3_ytrain)
ewa_score(d0v3.predict(d0v3_xtest),d0v3_ytest)


# #### D1V0 (1 sample)

# In[136]:


d1v0_output = np.float64(all_train_dfs['D1V0']['target'])


# #### D1V1 (33 samples)

# In[137]:


first = d1v1_df
d1v1_xtrain = first.iloc[:, 1:len(first.columns) - 1].values
d1v1_ytrain = first.loc[:, 'target'].values
# d1v1_xtest = second.iloc[:,1:len(second.columns)-1].values
# d1v1_ytest = second.loc[:,'target'].values
md = 3
d1v1 = DecisionTreeRegressor(max_depth=md, criterion='absolute_error', random_state=13740103)
d1v1.fit(d1v1_xtrain, d1v1_ytrain)


# #### D1V2 (1082 samples)

# In[138]:


#best solution
first = d1v2_df
d1v2_xtrain = first.iloc[:, 1:len(first.columns) - 1].values
d1v2_ytrain = first.loc[:, 'target'].values
# d1v2_xtest = second.iloc[:,1:len(second.columns)-1].values
# d1v2_ytest = second.loc[:,'target'].values
md = 11
d1v2 = DecisionTreeRegressor(max_depth=md, criterion='absolute_error', random_state=13740103)
d1v2.fit(d1v2_xtrain, d1v2_ytrain)


# #### D2V0 (13 samples)

# In[139]:


d2v0_output = d2v0_df['target'].mean()


# #### D2V1 (1092 samples)

# In[140]:


#best solution
first = d2v1_df
d2v1_xtrain = first.iloc[:, 1:len(first.columns) - 1].values
d2v1_ytrain = first.loc[:, 'target'].values
# d2v1_xtest = second.iloc[:,1:len(second.columns)-1].values
# d2v1_ytest = second.loc[:,'target'].values
md = 7
d2v1 = DecisionTreeRegressor(max_depth=md, criterion='absolute_error', random_state=13740103)
d2v1.fit(d2v1_xtrain, d2v1_ytrain)


# #### D3V0 (360 samples)

# In[141]:


d3v0_output = d3v0_df['target'].mean()


# ### Test

# #### Extracting base features for test data

# In[23]:


pv = test_base_features(pv_test)


# #### Dataframe split by number of defects and vacancies

# In[142]:


all_test_dfs = {}
for defect in range(0,4):
    for vacancy in range(0,4):
        new_df = generate_dataframe(pv, defect, vacancy)
        if len(new_df) != 0:
            all_test_dfs['D%sV%s'%(str(defect),str(vacancy))] = new_df


# #### Prepare features for D0V2

# In[143]:


#best solution
d0v2_df = all_test_dfs['D0V2']
one_two = []
for i in range(len(d0v2_df)):
    x1 = d0v2_df['vac1_x'].iloc[i]
    y1 = d0v2_df['vac1_y'].iloc[i]
    z1 = d0v2_df['vac1_z'].iloc[i]
    x2 = d0v2_df['vac2_x'].iloc[i]
    y2 = d0v2_df['vac2_y'].iloc[i]
    z2 = d0v2_df['vac2_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))

new_df = pd.DataFrame(columns=['id', 'vac1_type', 'vac2_type', 'one_two'])
new_df['id'] = d0v2_df['id']
new_df['vac1_type'] = d0v2_df['vac1_type']
new_df['vac2_type'] = d0v2_df['vac2_type']
new_df['one_two'] = one_two
d0v2_df = new_df


# #### Prepare features for D0V3

# In[144]:


d0v3_df = all_test_dfs['D0V3']
one_two = []
one_three = []
two_three = []
for i in range(len(d0v3_df)):
    x1 = d0v3_df['vac1_x'].iloc[i]
    y1 = d0v3_df['vac1_y'].iloc[i]
    z1 = d0v3_df['vac1_z'].iloc[i]
    x2 = d0v3_df['vac2_x'].iloc[i]
    y2 = d0v3_df['vac2_y'].iloc[i]
    z2 = d0v3_df['vac2_z'].iloc[i]
    x3 = d0v3_df['vac3_x'].iloc[i]
    y3 = d0v3_df['vac3_y'].iloc[i]
    z3 = d0v3_df['vac3_z'].iloc[i]

    vac_coords = [coord(x1,y1,z1),coord(x2,y2,z2),coord(x3,y3,z3)]
    
    diff_vectors = [
        coord(mo_x_range,0,0)+x_unit,
        coord(-mo_x_range,0,0)-x_unit,
        mo_y_dir+y_unit,
        coord(0,0,0)-mo_y_dir-y_unit,
        coord(mo_x_range,0,0)+x_unit+mo_y_dir+y_unit,
        coord(mo_x_range,0,0)+x_unit-mo_y_dir-y_unit,
        coord(-mo_x_range,0,0)-x_unit+mo_y_dir+y_unit,
        coord(-mo_x_range,0,0)-x_unit-mo_y_dir-y_unit,    
    ]
    
    all_one = [vac_coords[0]]
    all_two = [vac_coords[1]]
    all_three = [vac_coords[2]]
    for diff in diff_vectors:
        all_one.append(vac_coords[0]+diff)
        all_two.append(vac_coords[1]+diff)
        all_three.append(vac_coords[2]+diff)
    min_one_two = 1e5
    min_one_three = 1e5
    min_two_three = 1e5
    for pos in all_two:
        tmp1 = vac_coords[0].distance(pos)
        tmp2 = vac_coords[2].distance(pos)
        if tmp1 < min_one_two:
            min_one_two = tmp1
        if tmp2 < min_two_three:
            min_two_three = tmp2
    for pos in all_three:
        tmp = vac_coords[0].distance(pos)
        if tmp < min_one_three:
            min_one_three = tmp
    
    one_two.append(min_one_two)
    one_three.append(min_one_three)
    two_three.append(min_two_three)    

new_df = pd.DataFrame(
    columns=['id', 'one_two', 'one_three', 'two_three'])
new_df['id'] = d0v3_df['id']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
d0v3_df = new_df


# #### Prepare featurse for D1V1

# In[145]:


#best solution
d1v1_df = all_test_dfs['D1V1']
one_two = []
for i in range(len(d1v1_df)):
    x1 = d1v1_df['def1_x'].iloc[i]
    y1 = d1v1_df['def1_y'].iloc[i]
    z1 = d1v1_df['def1_z'].iloc[i]
    x2 = d1v1_df['vac1_x'].iloc[i]
    y2 = d1v1_df['vac1_y'].iloc[i]
    z2 = d1v1_df['vac1_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))

new_df = pd.DataFrame(columns=['id', 'vac1_type', 'def1_type', 'def1_ori', 'one_two'])
new_df['id'] = d1v1_df['id']
new_df['vac1_type'] = d1v1_df['vac1_type']
new_df['def1_type'] = d1v1_df['def1_type']
new_df['def1_ori'] = d1v1_df['def1_ori']
new_df['one_two'] = one_two
d1v1_df = new_df


# #### Prepare features for D1V2

# In[146]:


#best solution
d1v2_df = all_test_dfs['D1V2']
one_two = []
one_three = []
two_three = []
for i in range(len(d1v2_df)):
    x1 = d1v2_df['vac1_x'].iloc[i]
    y1 = d1v2_df['vac1_y'].iloc[i]
    z1 = d1v2_df['vac1_z'].iloc[i]
    x2 = d1v2_df['vac2_x'].iloc[i]
    y2 = d1v2_df['vac2_y'].iloc[i]
    z2 = d1v2_df['vac2_z'].iloc[i]
    x3 = d1v2_df['def1_x'].iloc[i]
    y3 = d1v2_df['def1_y'].iloc[i]
    z3 = d1v2_df['def1_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))
    one_three.append(distance_3d((x1, y1, z1), (x3, y3, z3)))
    two_three.append(distance_3d((x2, y2, z2), (x3, y3, z3)))

new_df = pd.DataFrame(
    columns=['id', 'num_bottom_vac', 'num_middle_vac', 'num_top_vac', 'vac1_type', 'vac2_type', 'def1_type', 'def1_ori',
             'one_two', 'one_three', 'two_three'])
new_df['id'] = d1v2_df['id']
new_df['num_bottom_vac'] = d1v2_df['num_bottom_vac']
new_df['num_middle_vac'] = d1v2_df['num_middle_vac']
new_df['num_top_vac'] = d1v2_df['num_top_vac']
new_df['vac1_type'] = d1v2_df['vac1_type']
new_df['vac2_type'] = d1v2_df['vac2_type']
new_df['def1_type'] = d1v2_df['def1_type']
new_df['def1_ori'] = d1v2_df['def1_ori']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
# new_df['target'] = d1v2_df['target']
d1v2_df = new_df


# #### Prepare features for D2V0

# In[147]:


#best solution
d2v0_df = all_test_dfs['D2V0']
one_two = []
for i in range(len(d2v0_df)):
    x1 = d2v0_df['def1_x'].iloc[i]
    y1 = d2v0_df['def1_y'].iloc[i]
    z1 = d2v0_df['def1_z'].iloc[i]
    x2 = d2v0_df['def2_x'].iloc[i]
    y2 = d2v0_df['def2_y'].iloc[i]
    z2 = d2v0_df['def2_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))

new_df = pd.DataFrame(columns=['id', 'def1_type', 'def1_ori', 'def2_type', 'def2_ori', 'one_two'])
new_df['id'] = d2v0_df['id']
new_df['def1_type'] = d2v0_df['def1_type']
new_df['def1_ori'] = d2v0_df['def1_ori']
new_df['def2_type'] = d2v0_df['def2_type']
new_df['def2_ori'] = d2v0_df['def2_ori']
new_df['one_two'] = one_two
d2v0_df = new_df


# #### Prepare features for D2V1

# In[148]:


#best solution
d2v1_df = all_test_dfs['D2V1']
one_two = []
one_three = []
two_three = []
for i in range(len(d2v1_df)):
    x1 = d2v1_df['vac1_x'].iloc[i]
    y1 = d2v1_df['vac1_y'].iloc[i]
    z1 = d2v1_df['vac1_z'].iloc[i]
    x2 = d2v1_df['def1_x'].iloc[i]
    y2 = d2v1_df['def1_y'].iloc[i]
    z2 = d2v1_df['def1_z'].iloc[i]
    x3 = d2v1_df['def2_x'].iloc[i]
    y3 = d2v1_df['def2_y'].iloc[i]
    z3 = d2v1_df['def2_z'].iloc[i]
    one_two.append(distance_3d((x1, y1, z1), (x2, y2, z2)))
    one_three.append(distance_3d((x1, y1, z1), (x3, y3, z3)))
    two_three.append(distance_3d((x2, y2, z2), (x3, y3, z3)))

new_df = pd.DataFrame(
    columns=['id', 'vac1_type', 'def1_type', 'def1_ori', 'def2_type', 'def2_ori', 'one_two', 'one_three', 'two_three'])
new_df['id'] = d2v1_df['id']
new_df['vac1_type'] = d2v1_df['vac1_type']
new_df['def1_type'] = d2v1_df['def1_type']
new_df['def1_ori'] = d2v1_df['def1_ori']
new_df['def2_type'] = d2v1_df['def2_type']
new_df['def2_ori'] = d2v1_df['def2_ori']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
d2v1_df = new_df


# #### Prepare features for D3V0

# In[149]:


d3v0_df = all_test_dfs['D3V0']
one_two = []
one_three = []
two_three = []
for i in range(len(d3v0_df)):
    x1 = d3v0_df['def1_x'].iloc[i]
    y1 = d3v0_df['def1_y'].iloc[i]
    z1 = d3v0_df['def1_z'].iloc[i]
    x2 = d3v0_df['def2_x'].iloc[i]
    y2 = d3v0_df['def2_y'].iloc[i]
    z2 = d3v0_df['def2_z'].iloc[i]
    x3 = d3v0_df['def3_x'].iloc[i]
    y3 = d3v0_df['def3_y'].iloc[i]
    z3 = d3v0_df['def3_z'].iloc[i]
    one_two.append(distance_3d((x1,y1,z1),(x2,y2,z2)))
    one_three.append(distance_3d((x1,y1,z1),(x3,y3,z3)))
    two_three.append(distance_3d((x2,y2,z2),(x3,y3,z3)))
new_df = pd.DataFrame(columns=['id','def1_type','def1_ori', 'def2_type','def2_ori','def3_type','def3_ori','one_two','one_three','two_three'])
new_df['id'] = d3v0_df['id']
new_df['def1_type'] = d3v0_df['def1_type']
new_df['def1_ori'] = d3v0_df['def1_ori']
new_df['def2_type'] = d3v0_df['def2_type']
new_df['def2_ori'] = d3v0_df['def2_ori']
new_df['def3_type'] = d3v0_df['def3_type']
new_df['def3_ori'] = d3v0_df['def3_ori']
new_df['one_two'] = one_two
new_df['one_three'] = one_three
new_df['two_three'] = two_three
d3v0_df = new_df


# ### Prediction

# #### Single samples predictions

# In[150]:


single_results = []
for key in all_test_dfs.keys():
    if len(all_test_dfs[key]) == 1:
        res = pd.DataFrame({'id':all_test_dfs[key]['id'],'predictions':np.float64(all_train_dfs[key]['target'])})
        single_results.append(res)


# #### D0V2

# In[151]:


d0v2_xtest = d0v2_df.iloc[:, 1:len(d0v2_df.columns)].values
d0v2_preds = d0v2.predict(d0v2_xtest)
res1 = pd.DataFrame({'id': d0v2_df['id'], 'predictions': d0v2_preds})


# #### D0V3

# In[152]:


d0v3_xtest = d0v3_df.iloc[:, 1:len(d0v3_df.columns)].values
d0v3_preds = d0v3.predict(d0v3_xtest)
res2 = pd.DataFrame({'id': d0v3_df['id'], 'predictions': d0v3_preds})


# #### D1V1

# In[153]:


d1v1_xtest = d1v1_df.iloc[:, 1:len(d1v1_df.columns)].values
d1v1_preds = d1v1.predict(d1v1_xtest)
res3 = pd.DataFrame({'id': d1v1_df['id'], 'predictions': d1v1_preds})


# #### D1V2

# In[154]:


d1v2_xtest = d1v2_df.iloc[:, 1:len(d1v2_df.columns)].values
d1v2_preds = d1v2.predict(d1v2_xtest)
res4 = pd.DataFrame({'id': d1v2_df['id'], 'predictions': d1v2_preds})


# #### D2V0

# In[155]:


d2v0_preds = [d2v0_output for i in range(len(d2v0_df))]
res5 = pd.DataFrame({'id': d2v0_df['id'], 'predictions': d2v0_preds})


# #### D2V1

# In[156]:


d2v1_xtest = d2v1_df.iloc[:, 1:len(d2v1_df.columns)].values
d2v1_preds = d2v1.predict(d2v1_xtest)
res6 = pd.DataFrame({'id': d2v1_df['id'], 'predictions': d2v1_preds})


# #### D3V0

# In[157]:


d3v0_preds = [d3v0_output for i in range(len(d3v0_df))]
res7 = pd.DataFrame({'id': d3v0_df['id'], 'predictions': d3v0_preds})


# ### Result

# In[158]:


res = res1.append(res2)
res = res.append(res3)
res = res.append(res4)
res = res.append(res5)
res = res.append(res6)
res = res.append(res7)
for i in single_results:
    res = res.append(i)
res.to_csv('./submission.csv', index=False)

