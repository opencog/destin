#ifndef VideoWriter_H
#define VideoWriter_H

#include "opencv/cv.h"
#include "opencv/highgui.h"

/** simple wrapper around OpenCV VideoWriter
  * suitable to be used with python bindings
  */
class VideoWriter {

    cv::VideoWriter writer;
    bool isOpened;
    std::string filename;
    float fps;
public:
    VideoWriter(std::string filename, float fps):
        isOpened(false), filename(filename),fps(fps){}

    void write(cv::Mat &img);
};

#endif
