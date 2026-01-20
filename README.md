# drop_diameter_postProcess
Python script to find diameter info from csv with (x,y,z,alpha)

Currently works with MFC and OpenFOAM. The OpenFOAM data needs to be postprocessed to be in a single csv per time step.

Tests currently don't work.

Known issues: doesn't change diameter values when the alpha threshold changes.
