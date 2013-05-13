# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:55:26 2013

@author: teaera
"""

import os
import time
import cv2.cv as cv
import pydestin as pd

import czt_mod as cm

#cm.processFld("/home/teaera/Work/RECORD/2013.5.8/org", "/home/teaera/Work/RECORD/2013.5.8/pro_1")

'''
dn = cm.init_destin(extRatio=1)
ims = pd.ImageSouceImpl()
#cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_2")
#cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_2_2")
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_only1")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, "/home/teaera/Pictures/2013_5_8_1.jpg")
'''


#############################################################################
# 2013.5.9
'''
start_time = time.time()
dn = cm.init_destin(extRatio=2)
cm.train_2flds(dn, "/home/teaera/Work/RECORD/2013.5.8/pro_3", "/home/teaera/Work/RECORD/2013.5.8/pro_add_3", 10)
end_time = time.time()
print("Cost: %s secs!" % (str(end_time-start_time)))
cm.dcis(dn, 7)
'''

#############################################################################
# 2013.5.10
'''
'''
size = 512*512
extRatio = 1
dn = cm.init_destin(extRatio=extRatio)
cm.cl.isNeedResize("/home/teaera/Work/RECORD/2013.5.8/pro_add_3/1.jpg")
f = cm.cl.get_float512()
cm.train_only(dn, f, 3000)
cm.dcis(dn, 7)