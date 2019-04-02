""" aimfuc.py """
import numpy as np

# ZDT1
def ZDT1(Chrom, LegV):
    ObjV1 = Chrom[:, 0]
    gx = 1 + (9 / 29) * np.sum(Chrom[:, 1:30], 1)
    hx = 1 - np.sqrt(ObjV1 / gx)
    ObjV2 = gx * hx

    return [np.array([ObjV1, ObjV2]).T, LegV]