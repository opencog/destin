#!/usr/bin/python
import pydestin as pd


def array_to_pointer(arr):
    ia = pd.SWIG_IntArray(len(arr))
    for x in range(len(arr)):
        ia[x] = arr[x]
    return ia


        
dn = pd.DestinNetworkAlt(pd.W512, 8, [2,2,2,2,2,2,2,2])

vs = pd.VideoSource(False, "IMG_1183.MOV")
vs.enableDisplayWindow()

def doFrame():
    if(vs.grab()):
        dn.doDestin(vs.getOutput())
        return True
    else:
        return False


