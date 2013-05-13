#include <cv.h>
#include <highgui.h>

using namespace std;

class czt_lib
{
private:
    cv::Mat currMat;
    cv::Mat floatMat;
    cv::Mat resizeMat;
    cv::Mat mat512;
    cv::Mat mat512_out;
    bool isResize;

public:
    void print1()
    {
        printf("This is czt_lib!\n");
    }

    bool isNeedResize(string imgPath)
    {
        cv::Mat inputIm = cv::imread(imgPath, 0); // CV_LOAD_IMAGE_GRAYSCALE failed! But 0 worked!!! Type: 8UC1
        currMat = inputIm;

        if(inputIm.cols == 512 && inputIm.rows == 512)
        {
            isResize = false;
        }
        else
        {
            isResize = true;
        }
        currMat.convertTo(floatMat, CV_32FC1, 1.0/255.0);
        if(isResize)
        {
            cv::resize(floatMat, resizeMat, cv::Size(512,512), 1.0, 1.0);
            mat512 = resizeMat;
        }
        else
        {
            mat512 = floatMat;
        }
        return isResize;
    }

    cv::Mat get_Mat512()
    {
        return mat512;
    }

    float * get_float512()
    {
        return (float *)mat512.data;
    }

    bool write_file(string file_path)
    {
        mat512.convertTo(mat512_out, CV_8UC1, 255);
        return cv::imwrite(file_path, mat512_out);
    }

    //*************************************************************************
    float * combineImgs(string file1, string file2)
    {
        cv::Mat tempMat1 = cv::imread(file1, 0);
        cv::Mat tempMat2 = cv::imread(file2, 0);
        float * tempFloat1 = (float *)tempMat1.data;
        float * tempFloat2 = (float *)tempMat2.data;

        // Testing for the whole size:
        //printf("%d\n", tempMat1.rows*tempMat1.cols+tempMat2.rows*tempMat2.cols);

        float * outFloat;
        int size = 512*512;
        int extRatio = 2;
        MALLOC(outFloat, float, size*extRatio);

        int i;
        for(i=0; i<size; ++i)
        {
            outFloat[i] = tempFloat1[i];
        }
        for(i=0; i<size; ++i)
        {
            outFloat[i+size] = tempFloat2[i];
        }

        return outFloat;
    }

    void test1()
    {
        float * f1;
        int i, size=32, width=4;
        MALLOC(f1, float, size);
        for(i=0; i<size; ++i)
        {
            f1[i] = (float)rand()/(float)RAND_MAX;
        }
        cv::Mat tempMat(4, 4, CV_32FC1, f1);
        cv::namedWindow("test1");
        cv::imshow("test1", tempMat);
        cv::waitKey(0);
    }
};
