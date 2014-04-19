#ifndef STEREOVISION_H
#define STEREOVISION_H


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
#include <vector>
#include <stdio.h>

#define RESULT_OK   0
#define RESULT_FAIL 1

class StereoVision
{
private:
    //chesboard board corners X,Y, N = X*Y ,  number of corners = (number of cells - 1)
    int cornersX,cornersY,cornersN;
    int sampleCount;
    bool calibrationStarted;
    bool calibrationDone;

    CvSize imageSize;
    int imageWidth;
    int imageHeight;

    vector<CvPoint2D32f> ponintsTemp[2];
    vector<CvPoint3D32f> objectPoints;
    vector<CvPoint2D32f> points[2];
    vector<int> npoints;

public:
    StereoVision(int imageWidth,int imageHeight);
    ~StereoVision();

    //matrices resulting from calibration (used for cvRemap to rectify images)
    CvMat *mx1,*my1,*mx2,*my2;

    CvMat* imagesRectified[2];
    CvMat  *imageDepth,*imageDepthNormalized;

    // 2013.4.29
    // CZT
    //
    CvMat * imageDepthNormalized_c1;

    void calibrationStart(int cornersX,int cornersY);
    int calibrationAddSample(IplImage* imageLeft,IplImage* imageRight);
    int calibrationEnd();

    int calibrationSave(const char* filename);
    int calibrationLoad(const char* filename);

    int stereoProcess(CvArr* imageSrcLeft,CvArr* imageSrcRight);

    CvSize getImageSize(){return imageSize;}
    bool getCalibrationStarted(){return calibrationStarted;}
    bool getCalibrationDone(){return calibrationDone;}
    int getSampleCount(){return sampleCount;}

};

#endif // STEREOVISION_H
