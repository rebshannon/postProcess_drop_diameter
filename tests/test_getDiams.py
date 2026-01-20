from getDiams import OpenFOAM_pv as OpenFOAM_pv
from getDiams import MFC as MFCode
import pytest
import numpy as np
import pandas as pd
import csv

@pytest.fixture
def OF():
    postProcFolder = "/postProcessing/pvData"
    timeStep = 1e-6
    meshDensity = 2e-6
    return OpenFOAM_pv(postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep)

@pytest.fixture
def MFC():
    postProcFolder = '/silo_hdf5'
    timeStep = 1e-6
    meshDensity = 0.0035/300
    return MFCode(postProcFolder=postProcFolder,meshDensity=meshDensity,timeStep=timeStep)

def test_find_matching_files(OF):
    directory = '/projectnb/aeracous/REBECCA/postProcessing/testingDirs/U267_D2_B3' + OF.postProcFolder
    known = ['/projectnb/aeracous/REBECCA/postProcessing/testingDirs/U267_D2_B3/postProcessing/pvData/cellCenterData_100.csv','/projectnb/aeracous/REBECCA/postProcessing/testingDirs/U267_D2_B3/postProcessing/pvData/cellCenterData_200.csv']
    found = OF.find_matching_files(directory)
    assert known == found

def test_get_time_from_fileName(OF):
    fileName = 'cellCenterData_100.csv'
    known = 100
    found = OF.get_time_from_fileName(fileName)
    assert found == known

def test_load_dataframe(OF):
    file = '/projectnb/aeracous/REBECCA/postProcessing/testingDirs/U267_D2_B3/postProcessing/pvData/cellCenterData_100.csv'
    #folder = 'U267_D2_B3'
    getData = True

    k_times = 0.0001 
    k_time_strs = 0.0001 
    k_df = pd.DataFrame({'Time':[0.000101,0.000101],'Cell Type':[12,12],'T':[338.6923,338.6921],'U:0':[109.8258,109.8277],\
            'U:1':[1.394629,1.400717],'U:2':[0,0],'alpha.water':[0,0],'cellCenterCoords:0':[-0.005,-0.005],\
            'cellCenterCoords:1':[0.000451,0.000453],'cellCenterCoords:2':[0,0],'p':[174088,174088],'p_rgh':[174088,174088],\
            'rho':[1.786606,1.786603]})
    times = []
    time_strs = []

    [f_df,f_times, f_time_strs] = OF.load_dataframe(file,getData,times,time_strs)

    errors = []
    if not np.isclose(k_times,f_times):
        errors.append("Times dataframe failed")
    if not np.isclose(k_time_strs, f_time_strs):
        errors.append("Time strings dataframe failed")
    if not (k_df == f_df).all:
            errors.append("Main dataframe failed")

    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_get_water_points(OF):
    df = pd.read_csv('/projectnb/aeracous/REBECCA/postProcessing/testingDirs/test_get_water_coords.csv')
    df_fail = pd.read_csv('/projectnb/aeracous/REBECCA/postProcessing/testingDirs/test_fail_get_water_coords.csv')
    OF.meshDensity = 0.022 / 200
    f_coords = OF.get_water_points(df)
    f_coords_fail = OF.get_water_points(df_fail)
 
    k_coords = np.array([[100,12,0,0.5],
                [10,5,0,1]])
    print(k_coords)
    print(f_coords)

    errors = []
    if not 0 == f_coords_fail:
        errors.append("Doesn't return zero for no water")
    if not (k_coords == f_coords).all():
        errors.append("Water coords wrong")
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

    
def test_calculate_diameters(OF):
    coords = pd.read_csv('/projectnb/aeracous/REBECCA/postProcessing/testingDirs/test_coords_for_diameters.csv')
    OF.meshDensity = 0.022/200
    [horizontal,vertical] = OF.calculate_diameters(coords.values)

    errors = []

    if not np.isclose(horizontal,0.021777386,rtol=1.1e-4):
        errors.append("Doesn't calculated horizontal diameter right")
    if not np.isclose(vertical,0.02173887172,rtol=1.1e-4):
        errors.append("Doesn't calculated vertical diameter right")
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_calculate_equator_diameter(OF):
    coords = pd.read_csv('/projectnb/aeracous/REBECCA/postProcessing/testingDirs/test_coords_for_diameters.csv')
    OF.meshDensity = 0.022/200
    f_equator = OF.calculate_equator_diameter(coords.values)

    k_equator = 0.021777386

    assert np.isclose(k_equator, f_equator)

def test_calculate_centOfMass_diameter(OF):
    coords = pd.read_csv('/projectnb/aeracous/REBECCA/postProcessing/testingDirs/test_coords_for_diameters.csv')
    OF.meshDensity = 0.022/200
    f_cent = OF.calculate_centOfMass_diameter(coords.values)

    k_cent = 0.02173887172

    assert np.isclose(k_cent, f_cent)

# def test_process_folder_diameter(OF):
#     OF.meshDensity = 2e-6
#     caseName = 'U267_D2_B1'
#     caseFolder = "/projectnb/aeracous/REBECCA/postProcessing/testingDirs/U267_D2_B1/postProcessing/pvData"
#     diameter_info = OF.process_folder_diameter(caseFolder,caseName)
    
#     k_diameterIndo = pd.read_csv('/projectnb/aeracous/REBECCA/postProcessing/testingDirs/test_process_diameter_results.csv')

#     assert np.allclose(k_diameterIndo, diameter_info)

def test_extract_and_combine_data(MFC):
    file = '/projectnb/aeracous/REBECCA/MFC/v5.0.6/shockDropParam/2D/case1/silo_hdf5/root/collection_0.silo'
    time = []
    time_strs = []
    folder = '/projectnb/aeracous/REBECCA/MFC/v5.0.6/shockDropParam/2D/case1/silo_hdf5/'
    tStep = 0

    f_alphaList = MFC.extract_and_combine_data(file,folder,tStep,time,time_strs)
    # how to get k_alphaList
