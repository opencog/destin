#include "stereocamera.h"

StereoCamera::StereoCamera()
{
    for(int lr=0;lr<2;lr++){
        captures[lr] = 0;
        frames[lr] = 0;
        framesGray[lr] = 0;
    }
    ready = false;
}

StereoCamera::~StereoCamera()
{
    for(int lr=0;lr<2;lr++){
        cvReleaseImage(&frames[lr]);
        cvReleaseImage(&framesGray[lr]);
        cvReleaseCapture(&captures[lr]);
    }

}

int StereoCamera::setup(CvSize imageSize){
    this->imageSize = imageSize;

    //captures[0] = cvCaptureFromCAM(CV_CAP_DSHOW + 0);
    //captures[1] = cvCaptureFromCAM(CV_CAP_DSHOW + 1);

    captures[0] = cvCaptureFromCAM(CV_CAP_ANY + 0);
    captures[1] = cvCaptureFromCAM(CV_CAP_ANY + 1);

    if(captures[0] == NULL)
    {
        printf("error\n");
    }

    if( captures[0] && captures[1]){

        for(int i=0;i<2;i++){
                cvSetCaptureProperty(captures[i], CV_CAP_PROP_FRAME_WIDTH, imageSize.width);
                cvSetCaptureProperty(captures[i], CV_CAP_PROP_FRAME_HEIGHT, imageSize.height);
        }

        printf("%d\n", (int)cvGetCaptureProperty(captures[0], CV_CAP_PROP_FRAME_WIDTH));
        printf("%d\n", (int)cvGetCaptureProperty(captures[0], CV_CAP_PROP_FRAME_HEIGHT));

        ready = true;
        return RESULT_OK;
    }else{
        ready = false;
        return RESULT_FAIL;
    }

}

int StereoCamera::capture(){
    frames[0] = cvQueryFrame(captures[0]);
    frames[1] = cvQueryFrame(captures[1]);
    return (captures[0] && captures[1]) ? RESULT_OK : RESULT_FAIL;
}

IplImage*  StereoCamera::getFramesGray(int lr){
    if(!frames[lr]) return 0;
    if(frames[lr]->depth == 1){
        framesGray[lr] = frames[lr];
        return frames[lr];
    }else{
        if(0 == framesGray[lr]) framesGray[lr] = cvCreateImage(cvGetSize(frames[lr]),IPL_DEPTH_8U,1);
        cvCvtColor(frames[lr],framesGray[lr],CV_BGR2GRAY);
        return framesGray[lr];
    }
}
