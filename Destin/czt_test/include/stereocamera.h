#ifndef STEREOCAMERA_H
#define STEREOCAMERA_H

#include "cv.h"
#include "cxmisc.h"
#include "cvaux.h"
#include "highgui.h"

#define RESULT_OK   0
#define RESULT_FAIL 1


class StereoCamera
{
    CvCapture* captures[2];

    CvSize imageSize;

public:
    IplImage* frames[2];
    IplImage* framesGray[2];


    StereoCamera();
    ~StereoCamera();
    int setup(CvSize imageSize);
    bool ready;
    int capture();
    IplImage* getFramesGray(int lr);

};

#endif // STEREOCAMERA_H
