#ifndef DST_IMAGE_SOUCE_IMPL_H
#define DST_IMAGE_SOUCE_IMPL_H


#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "ImageSourceBase.h"

using namespace std;

class ImageSouceImpl: public ImageSourceBase {
protected:

    virtual bool isImageIncluded(int index){
        return true;
    }

public:

    void addImage(string imagepath){
        cv::Mat grayIm = cv::imread(imagepath, 0);
        if(grayIm.data == NULL){
            printf("addImage: could not load image at %s\n", imagepath.c_str());
            return;
        }

        if(grayIm.type() == CV_8UC1){
            cv::Mat floatim;
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
        colorMats.push_back(colorIm);
        nImages++;
    }

};

#endif
