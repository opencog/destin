#include <cv.h>
#include <highgui.h>

using namespace std;

class czt_lib2
{
public:
    float * createFloatArr(int size)
    {
        float * outFloatArr;
        MALLOC(outFloatArr, float, size);
        return outFloatArr;
    }

    void combineInfo_extRatio(float * tempIn, int size, int extRatio, float * tempOut)
    {
        int i,j;
        for(i=0; i<size; ++i)
        {
            tempOut[i] = tempIn[i];
        }
        for(i=1; i<extRatio; ++i)
        {
            for(j=0; j<size; ++j)
            {
                tempOut[size*i+j] = (float)rand() / (float)RAND_MAX;
                //tempOut[size*i+j] = 0.9;
            }
        }
    }

    void combineInfo_depth(float * tempIn, float * depth, int size, float * tempOut)
    {
        int i;
        for(i=0; i<size; ++i)
        {
            tempOut[i] = tempIn[i];
        }
        for(i=0; i<size; ++i)
        {
            tempOut[size+i] = depth[i];
        }
    }

    //*************************************************************************
    void combineImgs(string file1, string file2, float * tempOut)
    {
        cv::Mat tempMat1 = cv::imread(file1, 0);
        cv::Mat tempMat2 = cv::imread(file2, 0);
        cv::Mat _tempMat1, _tempMat2;
        tempMat1.convertTo(_tempMat1, CV_32FC1, 1.0/255.0);
        tempMat2.convertTo(_tempMat2, CV_32FC1, 1.0/255.0);
        float * tempFloat1 = (float *)_tempMat1.data;
        float * tempFloat2 = (float *)_tempMat2.data;


        int i;
        int size = 512*512;
        for(i=0; i<size; ++i)
        {
            tempOut[i] = tempFloat1[i];
        }
        for(i=0; i<size; ++i)
        {
            tempOut[i+size] = tempFloat2[i];
        }
    }

    void getFloatFromImg(string file1, float * tempOut)
    {
        cv::Mat tempMat1 = cv::imread(file1, 0);
        cv::Mat _tempMat1;
        tempMat1.convertTo(_tempMat1, CV_32FC1, 1.0/255.0);
        tempOut = (float *)_tempMat1.data;
    }

    //*************************************************************************
    // 2013.5.27
    // template demo
    template <class T>
    T getMax(T a, T b)
    {
        return a>b?a:b;
    }
};
