#ifndef DST_IMAGE_SOUCE_IMPL_H
#define DST_IMAGE_SOUCE_IMPL_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ImageSourceBase.h"

using namespace std;

class ImageSouceImpl: public ImageSourceBase {

protected:

    /** Return true if the given image is allowed to be shown.
      * Allows subsclasses to enable or disable an image.
      * See CifarSouce::isImageIncluded
      */
    virtual bool isImageIncluded(int index){
        return true;
    }


public:

    ImageSouceImpl(int rows, int cols)
        :ImageSourceBase(rows, cols) {}


    virtual ~ImageSouceImpl(){}

    /**
      * Loads an image from a file from the given file path.
      */
    void addImage(string imagepath){
        cv::Mat grayIm = cv::imread(imagepath, 0);
        if(grayIm.data == NULL){
            printf("addImage: could not load image at %s\n", imagepath.c_str());
            return;
        }

        if(grayIm.type() == CV_8UC1){
            cv::Mat floatim;
            // convert from grayscale byte range 0 -> 255 to float range 0.0 -> 1.0
            grayIm.convertTo(floatim, CV_32FC1, 1.0/255.0);
            grayMats.push_back(floatim);
        }else{
            throw std::runtime_error("ImageSouceImpl::addImage(int) unsupported opencv grayscale image type\n.");
        }

        cv::Mat colorIm = cv::imread(imagepath,1);
        if(colorIm.data == NULL){
            printf("addImage: could not load color image at %s\n", imagepath.c_str());
            return;
        }
        if(colorIm.type() != CV_8UC3){
            stringstream ss; ss << __PRETTY_FUNCTION__ << ", unexpected opencv color image type." ;
            throw std::runtime_error(ss.str());
        }
        colorMats.push_back(colorIm);

        addColorFloatImage(colorIm);

        nImages++;
    }
};

#endif
