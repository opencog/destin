#include "stereovision.h"

StereoVision::StereoVision(int imageWidth,int imageHeight)
{
    this->imageSize = cvSize(imageWidth,imageHeight);
    mx1 = my1 = mx2 = my2 = 0;
    calibrationStarted = false;
    calibrationDone = false;
    imagesRectified[0] = imagesRectified[1] = imageDepth = imageDepthNormalized = 0;
    imageDepth = 0;
	sampleCount = 0;

    // 2013.4.29
    // CZT
    //
    imageDepthNormalized_c1 = 0;
}

StereoVision::~StereoVision()
{
    cvReleaseMat(&imagesRectified[0]);
    cvReleaseMat(&imagesRectified[1]);
    cvReleaseMat(&imageDepth);
    cvReleaseMat(&imageDepthNormalized);
}

void StereoVision::calibrationStart(int cornersX,int cornersY){
    this->cornersX = cornersX;
    this->cornersY = cornersY;
    this->cornersN = cornersX*cornersY;
    ponintsTemp[0].resize(cornersN);
    ponintsTemp[1].resize(cornersN);
    sampleCount = 0;
    calibrationStarted = true;
}


int StereoVision::calibrationAddSample(IplImage* imageLeft,IplImage* imageRight){

    if(!calibrationStarted) return RESULT_FAIL;

    IplImage* image[2] = {imageLeft,imageRight};

    int succeses = 0;

    for(int lr=0;lr<2;lr++){
        CvSize imageSize =  cvGetSize(image[lr]);

        if(imageSize.width != this->imageSize.width || imageSize.height != this->imageSize.height)
            return RESULT_FAIL;

        int cornersDetected = 0;

        //FIND CHESSBOARDS AND CORNERS THEREIN:
        int result = cvFindChessboardCorners(
            image[lr], cvSize(cornersX, cornersY),
            &ponintsTemp[lr][0], &cornersDetected,
            CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE
        );

        if(result && cornersDetected == cornersN){


            //Calibration will suffer without subpixel interpolation
            cvFindCornerSubPix(
                image[lr], &ponintsTemp[lr][0], cornersDetected,
                cvSize(11, 11), cvSize(-1,-1),
                cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30, 0.01)
            );
            succeses++;
        }

    }
    if(2==succeses){
        for(int lr=0;lr<2;lr++){
            points[lr].resize((sampleCount+1)*cornersN);
            copy( ponintsTemp[lr].begin(), ponintsTemp[lr].end(),  points[lr].begin() + sampleCount*cornersN);
        }
        sampleCount++;
        return RESULT_OK;
    }else{
        return RESULT_FAIL;
    }
}

int StereoVision::calibrationEnd(){
    calibrationStarted = false;

    // ARRAY AND VECTOR STORAGE:
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    CvMat _M1,_M2,_D1,_D2,_R,_T,_E,_F;

    _M1 = cvMat(3, 3, CV_64F, M1 );
    _M2 = cvMat(3, 3, CV_64F, M2 );
    _D1 = cvMat(1, 5, CV_64F, D1 );
    _D2 = cvMat(1, 5, CV_64F, D2 );
    _R = cvMat(3, 3, CV_64F, R );
    _T = cvMat(3, 1, CV_64F, T );
    _E = cvMat(3, 3, CV_64F, E );
    _F = cvMat(3, 3, CV_64F, F );

    // HARVEST CHESSBOARD 3D OBJECT POINT LIST:
    objectPoints.resize(sampleCount*cornersN);

    for(int k=0;k<sampleCount;k++)
        for(int i = 0; i < cornersY; i++ )
            for(int j = 0; j < cornersX; j++ )
                objectPoints[k*cornersY*cornersX + i*cornersX + j] = cvPoint3D32f(i, j, 0);


    npoints.resize(sampleCount,cornersN);

    int N = sampleCount * cornersN;


    CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
    CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
    cvSetIdentity(&_M1);
    cvSetIdentity(&_M2);
    cvZero(&_D1);
    cvZero(&_D2);

    //CALIBRATE THE STEREO CAMERAS
    cvStereoCalibrate( &_objectPoints, &_imagePoints1,
        &_imagePoints2, &_npoints,
        &_M1, &_D1, &_M2, &_D2,
        imageSize, &_R, &_T, &_E, &_F,
        cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
        CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_ZERO_TANGENT_DIST + CV_CALIB_SAME_FOCAL_LENGTH
    );

    //Always work in undistorted space
    cvUndistortPoints( &_imagePoints1, &_imagePoints1,&_M1, &_D1, 0, &_M1 );
    cvUndistortPoints( &_imagePoints2, &_imagePoints2,&_M2, &_D2, 0, &_M2 );

    //COMPUTE AND DISPLAY RECTIFICATION


    double R1[3][3], R2[3][3];
    CvMat _R1 = cvMat(3, 3, CV_64F, R1);
    CvMat _R2 = cvMat(3, 3, CV_64F, R2);

    //HARTLEY'S RECTIFICATION METHOD
    double H1[3][3], H2[3][3], iM[3][3];
    CvMat _H1 = cvMat(3, 3, CV_64F, H1);
    CvMat _H2 = cvMat(3, 3, CV_64F, H2);
    CvMat _iM = cvMat(3, 3, CV_64F, iM);

    cvStereoRectifyUncalibrated(
        &_imagePoints1,&_imagePoints2, &_F,
        imageSize,
        &_H1, &_H2, 3
    );
    cvInvert(&_M1, &_iM);
    cvMatMul(&_H1, &_M1, &_R1);
    cvMatMul(&_iM, &_R1, &_R1);
    cvInvert(&_M2, &_iM);
    cvMatMul(&_H2, &_M2, &_R2);
    cvMatMul(&_iM, &_R2, &_R2);


    //Precompute map for cvRemap()
    cvReleaseMat(&mx1);
    cvReleaseMat(&my1);
    cvReleaseMat(&mx2);
    cvReleaseMat(&my2);
    mx1 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
    my1 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
    mx2 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
    my2 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );

    cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_M1,mx1,my1);
    cvInitUndistortRectifyMap(&_M2,&_D2,&_R2,&_M2,mx2,my2);

    calibrationDone = true;

    return RESULT_OK;
}

int StereoVision::stereoProcess(CvArr* imageSrcLeft,CvArr* imageSrcRight){
    if(!calibrationDone) return RESULT_FAIL;
    if(!imagesRectified[0]) imagesRectified[0] = cvCreateMat( imageSize.height,imageSize.width, CV_8U );
    if(!imagesRectified[1]) imagesRectified[1] = cvCreateMat( imageSize.height,imageSize.width, CV_8U );
    if(!imageDepth) imageDepth = cvCreateMat( imageSize.height,imageSize.width, CV_16S );
    if(!imageDepthNormalized) imageDepthNormalized = cvCreateMat( imageSize.height,imageSize.width, CV_8U );

    // 2013.4.29
    // CZT
    //
    if(!imageDepthNormalized_c1) imageDepthNormalized_c1 = cvCreateMat( 512, 512, CV_8U );

    /*if(!imagesRectified[0]) imagesRectified[0] = cvCreateMat( 512, 512, CV_8U );
    if(!imagesRectified[1]) imagesRectified[1] = cvCreateMat( 512, 512, CV_8U );
    if(!imageDepth) imageDepth = cvCreateMat( 512, 512, CV_16S );
    if(!imageDepthNormalized) imageDepthNormalized = cvCreateMat( 512, 512, CV_8U );*/

    //rectify images
    cvRemap( imageSrcLeft, imagesRectified[0] , mx1, my1 );
    cvRemap( imageSrcRight, imagesRectified[1] , mx2, my2 );


    CvStereoBMState *BMState = cvCreateStereoBMState();
    BMState->preFilterSize=41;
    BMState->preFilterCap=31;
    BMState->SADWindowSize=41;
    BMState->minDisparity=-64;
    BMState->numberOfDisparities=128;
    BMState->textureThreshold=10;
    BMState->uniquenessRatio=15;

    cvFindStereoCorrespondenceBM( imagesRectified[0], imagesRectified[1], imageDepth, BMState);

    cvNormalize( imageDepth, imageDepthNormalized, 0, 255, CV_MINMAX );

    cvReleaseStereoBMState(&BMState);

    // 2013.4.29
    // CZT
    //
    cvResize(imageDepthNormalized, imageDepthNormalized_c1);
    cvFlip(imageDepthNormalized_c1, imageDepthNormalized_c1, 1); // Flip, y-axis

    return RESULT_OK;
}

int StereoVision::calibrationSave(const char* filename){
    if(!calibrationDone) return RESULT_FAIL;
    FILE* f =  fopen(filename,"wb");
    if(!f) return RESULT_FAIL;
    if(!fwrite(mx1->data.fl,sizeof(float),mx1->rows*mx1->cols,f)) return RESULT_FAIL;
    if(!fwrite(my1->data.fl,sizeof(float),my1->rows*my1->cols,f)) return RESULT_FAIL;
    if(!fwrite(mx2->data.fl,sizeof(float),mx2->rows*mx2->cols,f)) return RESULT_FAIL;
    if(!fwrite(my2->data.fl,sizeof(float),my2->rows*my2->cols,f)) return RESULT_FAIL;
    fclose(f);
    return RESULT_OK;
}


int StereoVision::calibrationLoad(const char* filename){
    cvReleaseMat(&mx1);
    cvReleaseMat(&my1);
    cvReleaseMat(&mx2);
    cvReleaseMat(&my2);
    mx1 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
    my1 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
    mx2 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
    my2 = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
    FILE* f =  fopen(filename,"rb");
    if(!f) return RESULT_FAIL;
    if(!fread(mx1->data.fl,sizeof(float),mx1->rows*mx1->cols,f)) return RESULT_FAIL;
    if(!fread(my1->data.fl,sizeof(float),my1->rows*my1->cols,f)) return RESULT_FAIL;
    if(!fread(mx2->data.fl,sizeof(float),mx2->rows*mx2->cols,f)) return RESULT_FAIL;
    if(!fread(my2->data.fl,sizeof(float),my2->rows*my2->cols,f)) return RESULT_FAIL;
    fclose(f);
    calibrationDone = true;
    return RESULT_OK;
}

