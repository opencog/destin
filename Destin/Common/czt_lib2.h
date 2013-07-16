#include <cv.h>
#include <highgui.h>

using namespace std;

class czt_lib2
{
public:

    void free_f1dim(float * input)
    {
        if(input != NULL)
        {
            free(input);
        }
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

    // 2013.6.25
    void combineBGR(float * b, float * g, float * r, int size, float * out)
    {
        int i;
        for(i=0; i<size; ++i)
        {
            out[i] = b[i];
            out[size+i] = g[i];
            out[size+size+i] = r[i];
        }
    }

    //*************************************************************************
    void convert(cv::Mat & in, float * out) {
        if(in.channels()!=1){
            throw runtime_error("Excepted a grayscale image with one channel.");
        }
        if(in.depth()!=CV_8U){
            throw runtime_error("Expected image to have bit depth of 8bits unsigned integers ( CV_8U )");
        }
        cv::Point p(0, 0);
        int i = 0 ;
        for (p.y = 0; p.y < in.rows; p.y++) {
            for (p.x = 0; p.x < in.cols; p.x++) {
                //i = frame.at<uchar>(p);
                //use something like frame.at<Vec3b>(p)[channel] in case of trying to support color images.
                //There would be 3 channels for a color image (one for each of r, g, b)

                //out[i] = (float)in.at<uchar>(p) / 255.0f;
                in.at<float>(p) = (float)(out[i]*255.0f);
                i++;
            }
        }
    }

    float * floatArrCreate(int size)
    {
        float * outFloatArr;
        outFloatArr = (float *)malloc(size * sizeof(float));
        return outFloatArr;
    }

    void floatArrRandomize(float * inArr, int size)
    {
        int i;
        for(i=0; i<size; ++i)
        {
            inArr[i] = (float)rand() / (float)RAND_MAX;
        }
    }

};
