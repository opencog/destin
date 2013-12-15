#include "VideoWriter.h"

void VideoWriter::write(cv::Mat &img){
    if(!isOpened){

        bool opened = writer.open(filename, CV_FOURCC('M','P','4','V'), fps, cv::Size(img.size()), true);
        if(!opened){
            throw std::runtime_error("could not open file for video writting");
        }
        isOpened = true;
    }
    writer.write(img);
}
