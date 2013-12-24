#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:00:09 2013

@author: ted
"""

import unittest
from heirarchy_builder import *

import common as cm
import pydestin as pd

class TestHeirarchyBuilder(unittest.TestCase):
    def test_bad_input_dim(self):
        s = self
        
        hb = HeirarchyBuilder()
        hb.build_classic(1, 5)
        
        s.assertRaises(Exception, hb.build_classic, hb, 2, 5)
        
    def test_DestinNetworkAlt(self):
        layer_widths=[32, 16, 8, 7, 6, 5, 4, 3, 2, 1]
        centroids=   [2,   8, 32,32,64,16,16,16,16,4]
        network = pd.DestinNetworkAlt(pd.W512,len(layer_widths), centroids, True, layer_widths, pd.DST_IMG_MODE_GRAYSCALE)
        node = pd.GetNodeFromDestin(network.getNetwork(), 0, 1, 1)
        self.assertEqual(1, node.nParents)
        node = pd.GetNodeFromDestin(network.getNetwork(), 2, 1, 1)
        self.assertEqual(4, node.nParents)
        
    def test_common_init(self):
        layer_widths=[32, 16, 8, 7, 6, 5, 4, 3, 2, 1]
        centroids=   [2,   8, 32,32,64,16,16,16,16,4]
        cm.init(centroids=centroids,
                video_file="moving_square.avi",
                learn_rate=0.05,
                layer_widths=layer_widths,
                img_width=512
                )
        
    def test_tree_sizes(self):
        layer_widths=[32, 16, 8, 7, 6, 5, 4, 3, 2, 1]
        centroids=   [2,   8, 32,32,64,16,16,16,16,4]
        network = pd.DestinNetworkAlt(pd.W512,len(layer_widths), centroids, True, layer_widths, pd.DST_IMG_MODE_GRAYSCALE)
        dtm = pd.DestinTreeManager(network, 0)
        print "Size is:" + str(dtm.getWinningCentroidTreeSize())

def main():
    unittest.main()

if __name__ == '__main__':
    main()
