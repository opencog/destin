#ifndef GENERATIVE_VISUALIZER_H
#define GENERATIVE_VISUALIZER_H

#include <math.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "destin.h"

class GenerativeVisualizer {

private:
    Destin * destin;

    uint lastImageWidth;

    cv::Mat image;

    void visualize(int layer, int row, int col){

        //calculate image size from layer #
        Node * n = GetNodeFromDestin(destin, 0, 0, 0);
        uint sqWidth = sqrt(n->ni);
        uint imageWidth = sqWidth * pow(2, layer );
        CvSize size;
        size.width = imageWidth;
        size.height = imageWidth;
        if(imageWidth != lastImageWidth){
            //different image size so allocate a new image
            //c++ should take care of memory management
            image = cv::Mat(size, CV_32FC1);
        }

        lastImageWidth = imageWidth;

        GenerateInputFromBelief(destin, (float *) image.data );
        cv::imshow("Generative Image", image);
        cv::waitKey();
    }

public:
    GenerativeVisualizer(Destin * d)
        :destin(d), lastImageWidth(0){}

    void visualize(){
        visualize(destin->nLayers - 1, 0, 0);
    }

    /**
    * Opens a window to show a gradient grayscale image
    */
    void testShowImage(){
        cv::Mat img(512,512, CV_32FC1);
        //grayscale float values between 0.0 and 1.0
        //0.0 is black, 1.0 is white
        float  * data = (float *)img.data;
        for(int i  = 0 ; i < 512 * 512 ; i++)
        {
            data[i] = i / (float)(512 * 512);
        }
        cv::imshow("a picture", img);
        cv::waitKey();

    }

};

#endif
