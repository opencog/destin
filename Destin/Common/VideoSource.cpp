/*
 * VideoSource.cpp
 *
 *  Created on: Nov 3, 2011
 *      Author: ted
 */

#include "VideoSource.h"
#include <iostream>

using namespace std;


void VideoSource::convert(cv::Mat & in, float * out) {

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
            out[i] = (float)in.at<uchar>(p) / 255.0f;
            i++;
        }
    }
}


/**
 * @brief Retrieve, transform and show the image.
 */
void VideoSource::processFrame(){
    cap->retrieve(original_frame); //retrieve the captured frame

    original_size = original_frame.size();

    if(original_size == target_size){
        rescaled_frame = original_frame;
    }else{
        // The camera may not deliver the size you ask for. It may
        // try to find the closest resolution it can.
        // So we have to resize it manually to ge the exact size we want.
        // see http://opencv.willowgarage.com/documentation/cpp/imgproc_geometric_image_transformations.html#resize
        cv::resize(original_frame, rescaled_frame, target_size, 1.0, 1.0);
    }

    if(flip){
        cv::flip(rescaled_frame, flipped_frame, 1);
    } else {
        flipped_frame = rescaled_frame;
    }

    cvtColor(flipped_frame, greyscaled_frame, CV_BGR2GRAY); //turn the image grey scale

    //convert the greyscaled_frame into a float array for DeSTIN
    convert(greyscaled_frame, this->float_frame);

    if(edge_detection){
        cv::GaussianBlur(greyscaled_frame, greyscaled_frame, cv::Size(7,7) , .75, .75); // blur the image
        cv::Canny(greyscaled_frame, greyscaled_frame, 0, 30, 3); // apply edge detection
    }

    // 2013.6.25
    if(isShowColor){
        splitBGR();
    }

    if(showWindow){
        if(!isShowColor){
			cv::imshow(this->win_title, greyscaled_frame); //show video to output window
		}else{
			cv::imshow(this->win_title, flipped_frame);
		}
    }
    return;
}

bool VideoSource::grab(){

    // if a video, rewind it if it's at the end of the video.
    if(!isDevice){
        int currentFrame = (int)cap->get(CV_CAP_PROP_POS_FRAMES);
        int totalFrames = (int)cap->get(CV_CAP_PROP_FRAME_COUNT);

        // rewind if at the end of the video
        if(currentFrame >= totalFrames){
            cap->set(CV_CAP_PROP_POS_FRAMES,0);
        }
    }


    if(cap->grab()){
        processFrame(); // Retrieve, transform and show the image.
    } else {
        cerr << "Could not grab the frame." << endl;
        return false;
    }

    // Some strange issues with waitkey, see http://opencv.willowgarage.com/wiki/documentation/c/highgui/WaitKey
    // waitKey needs to be called to give the OS time to update drawn images.
    // If the time given is too small, the images wont be updated at all.
    // If the time given is too high, then destin frames per seconds drops.
    // Waitkey also returns a value other than 0 if a key is pressed.
    if(showWindow && cv::waitKey(5)>=0){
        cerr << "Key pressed, stopping video." << endl;
        return false;
    }
    return true;
}

bool VideoSource::rewind(){
     if(cap->get(CV_CAP_PROP_FRAME_COUNT) == 0){
         return true; //already at the begining
     }
     cap->set(CV_CAP_PROP_POS_FRAMES,0);
     return true;
}
