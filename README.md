# Post Processing Scripts for Shock Droplet Interactions

Python scripts that extract droplet dimeters, leading edge, and perimeter from CFD output. Compatible with OpenFOAM and MFC.

OpenFOAM does not have perimeter capability yet.

Adapted from Brendon's work.


## MFC
Define
1. working directory
2. post processing directory
3. Number of processors used
4. desired alpha threshold
5. simulation time step (need to confirm if this is used)
6. Mesh density

working directory + case list + post processing directory = path to where the data is

## OpenFAOM
Define
1. working directory
2. desired alpha threshold
3. simulation time step (need to confirm if this is used)
4. Mesh density

## For all cases
1. choose is you want perimeter, diameter, or both
2. make case list selection

## To Run
`source  /p/home/rebshan/postProcessing/myenv/bin/activate`

`python run_\<fileName\>`

## OpenFAOM pv (to be deprecated)
or potenailly updated to be used with a csv

Define
1. working directory
2. post processing directory
3. desired alpha threshold
4. simulation time step (need to confirm if this is used)
5. Mesh density

working directory + case list + post processing directory = path to where the data is

In src getDiams_OF_pv ensure:

pattern matches the file names for pvData csvs

alphaVar, x, y, and z match column headers to pvData csvs

