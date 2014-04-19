#ifndef VideoWriter_H
#define VideoWriter_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>

/** A Simple wrapper around OpenCV VideoWriter.
  * so it can be used easily with Python bindings.
  */
class VideoWriter {

    cv::VideoWriter writer;
    bool isOpened;
    std::string filename;
    float fps;
public:
    VideoWriter(std::string filename, float fps):
        isOpened(false), filename(filename),fps(fps){}

    /** Write a video frame to the movie file.
     * Opens the video file if not open already.
     * Uses
     * @brief write
     * @param img - opencv image to write to the video
     */
    void write(cv::Mat &img);
};

#endif
