#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

class CztMod
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

    void combineFloatArrays(float * tempIn, float * tempIn2, int size, float * tempOut)
    {
        int i;
        for(i=0; i<size; ++i)
        {
            tempOut[i] = tempIn[i];
        }
        for(i=0; i<size; ++i)
        {
            tempOut[size+i] = tempIn2[i];
        }
    }

    //*************************************************************************
    // Combine 2 512*512 images into a float array with size 512*512*2
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
        int size = 512*512;  // Locked size here...
        for(i=0; i<size; ++i)
        {
            tempOut[i] = tempFloat1[i];
        }
        for(i=0; i<size; ++i)
        {
            tempOut[i+size] = tempFloat2[i];
        }
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
    // Convert a grayscale image into a float array
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

    // Create a float array with the given size
    float * floatArrCreate(int size)
    {
        float * outFloatArr;
        outFloatArr = (float *)malloc(size * sizeof(float));
        return outFloatArr;
    }

    // Randomize the float array with the given size
    void floatArrRandomize(float * inArr, int size)
    {
        int i;
        for(i=0; i<size; ++i)
        {
            inArr[i] = (float)rand() / (float)RAND_MAX;
        }
    }

    // Resize image and save the result, according to the size;
    void resizeImage(string imgPath, string savePath, cv::Size size)
    {
        cv::Mat inputMat = cv::imread(imgPath, 0); // CV_LOAD_IMAGE_GRAYSCALE failed! But 0 worked!!! Type: 8UC1
        cv::Mat tempMat;
        cv::Mat resizeMat;

        inputMat.convertTo(tempMat, CV_32FC1, 1.0/255.0);
        cv::resize(tempMat, resizeMat, size, 1.0, 1.0);
        resizeMat.convertTo(tempMat, CV_8UC1, 255);

        if(savePath != "")
        {
            cv::imwrite(savePath, tempMat);
        }
    }

    //*************************************************************************
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

    float * get_float512()
    {
        return (float *)mat512.data;
    }

private:
    cv::Mat currMat;
    cv::Mat floatMat;
    cv::Mat resizeMat;
    cv::Mat mat512;
    cv::Mat mat512_out;
    bool isResize;

};
