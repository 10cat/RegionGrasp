import os
import vtk


def loadPLY(filenamePLY):
    readerPLY = vtk.vtkPLYReader()
    readerPLY.SetFileName(filenamePLY)
    readerPLY.Update()
    polydata = readerPLY.GetOutput() # get the vtkpolydata form

    if polydata.GetNumberOfPoints() == 0:
        raise ValueError(
            "No point data could be loaded from '" + filenamePLY)
        return None

