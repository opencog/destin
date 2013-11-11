# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:10:41 2013

@author: ted
"""
import unittest

class HeirarchyBuilder:
    """ Provides methods to specify the destin heirarchy """
    
    def __init__(self):
        self.layers = 0;
        self.centroids = [];
        self.layer_widths = [];
        self.image_width = 0;
        self.input_dim = 0;
            
    def build_classic(self, layers, image_width):
        self.layers = layers
        self.image_width = image_width

        for i in xrange(layers):
            self.layer_widths.append(2**(layers - i - 1))
            
        bottom_width = self.layer_widths[0]
        input_dim = image_width * image_width /  float(bottom_width * bottom_width)
        
        if input_dim != int(input_dim):
           raise Exception("image_width does not match layers")
            
        return


    def hybrid(self, pixel_region_width, img_width, non_overlap_count):
        if img_width % pixel_region_width != 0:
            raise Exception("pixel_region_width is not compatible with img_width!")
    
        self.layer_widths = []
    
        layer_width = img_width / pixel_region_width
        self.layer_widths.append(layer_width)
    
        for i in xrange(non_overlap_count):
            layer_width = layer_width / 2
            self.layer_widths.append(layer_width);
            if layer_width == 1:
                return self.layer_widths
    
        for i in xrange(layer_width - 1, 0, -1):
            self.layer_widths.append(i)
    
        print self.layer_widths
        return self.layer_widths

class TestHeirarchyBuilder(unittest.TestCase):
    def test_bad_input_dim(self):
        s = self
        
        hb = HeirarchyBuilder()
        hb.build_classic(1, 5)
        
        s.assertRaises(Exception, hb.build_classic, hb, 2, 5)
        
        
        


def main():
    unittest.main()

if __name__ == '__main__':
    main()
    