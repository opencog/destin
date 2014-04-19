/*
 * VideoSource.h
 *
 *  Created on: Nov 3, 2011
 *      Author: Ted Sanders
 */

#ifndef VIDEOSOURCE_H_
#define VIDEOSOURCE_H_


#include <stdexcept>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifndef _WIN32
extern "C"{
#define UINT64_C //hack to avoid compile error in libavutil/log.h
#include <libavutil/log.h> //used to turn off warning message
//"No accelerated colorspace conversion found from yuv422p to bgr24."
// that occurs when opening certain video files
}
#endif
using namespace std;


class VideoSource {
private:

	//webcam output is resized to this size:
	cv::Size target_size;

	//pointer to the video source
	cv::VideoCapture * cap;

	//video as originally comes out of the webcam
	cv::Mat original_frame;
    //video after it has been resized to the target_size
    cv::Mat rescaled_frame;
	//video after it has been resize and converted to greyscale
	cv::Mat greyscaled_frame;
    //video after it has been flip flipped
    cv::Mat flipped_frame;

	//the greyscaled_frame converted to a float array to be fed to DeSTIN
	float * float_frame;

	//the original size as it came out of the video device
	cv::Size original_size;

    // 2013.4.19
    // CZT
    // Defined title for windows:
    string win_title;

    // 2013.6.25
    float * bFrame;
    float * gFrame;
    float * rFrame;
    bool isShowColor;

	bool edge_detection; //if true shows video in edge detection mode ( shows image outlines)

    bool showWindow; //shows video or webcam input in window for user. 

    bool flip;      //

    bool isDevice;  // true if using webcam, otherwise using video file as the source.


	/**
	 * convert - converts from an OpenCV Mat greyscal 8bit uchar image into
     * a float array where each element is normalized to 0 to 1.0, with 1.0 being black.
	 *
	 * Assumes out points to a preallocated float array big enough to hold the
	 * converted image ( should be of size target_size)
	 */
    void convert(cv::Mat & in, float * out);

    void processFrame();

public:
	/**
	 * use_device - if true then will find the default device i.e. webcam as input
	 * and will ignore video_file, otherwise use the video file at video_file path as input
	 *
	 * video_file - video file to use as input when use_device is false
	 *
	 * dev_no - if use_device is true, specifies the device number to use,
	 * a value of 0 should bring up the default device.
	 *
	 */
	VideoSource(bool use_device, std::string video_file, int dev_no = 0) :
        target_size(512, 512), edge_detection(false), showWindow(false),
        isShowColor(false), flip(true), isDevice(use_device),
        win_title("DeSTIN Input Video") {

		float_frame = new float[target_size.area()];
		stringstream mess;
		if (use_device) {
			cap = new cv::VideoCapture(dev_no);
		} else {
			cap = new cv::VideoCapture(video_file);
		}
		if(!cap->isOpened()){
			if(use_device){
				mess << "Could not open capturing device with device number " << dev_no << endl;
			}else{
				mess << "Could not open video file " << video_file << endl;
			}
			throw runtime_error(mess.str());
		}

        /*cap->set(CV_CAP_PROP_FRAME_WIDTH, target_size.width);
        cap->set(CV_CAP_PROP_FRAME_HEIGHT, target_size.height);*/
        cvMoveWindow(win_title.c_str(), 50, 50);
#ifndef _WIN32
        av_log_set_level(AV_LOG_QUIET);//turn off message " No accelerated colorspace conversion found from yuv422p to bgr24"
#endif
	}

	bool isOpened() {
		return cap->isOpened();
    }

    /** Sets the size of the video output.
     * If the webcam can't supply the exact size image, it may
     * supply the resolution that is closest but smaller than what is asked.
     * In that case, the software will manually upscale the image the
     * rest of the way to match the exact size.
	 */
	void setSize(int width, int height) {
		if(target_size.width != width || target_size.height != height){
			delete [] float_frame;
			float_frame = new float[width * height];
		}
		target_size = cv::Size(width, height);
        cap->set(CV_CAP_PROP_FRAME_WIDTH, width);
        cap->set(CV_CAP_PROP_FRAME_HEIGHT, height);
	}
   

    /**
        If true, then the video is flipped on the vertical axis ( mirror flipped).
        Defaults to true.
    */
    void setFlip(bool isFlipped){
        this->flip = isFlipped;
    }

	~VideoSource() {
		delete cap;
		delete [] float_frame;

        if(isShowColor)
        {
            delete [] bFrame;
            delete [] gFrame;
            delete [] rFrame;
        }
	}


	/**
	 * Gets the pointer to the greyscaled video image to
	 * be fed to DeSTIN. This pointer should not be deleted.
	 * Points to a float array of length width*height.
	 * Pixel values are between 0 and 1.
	 * 0.0 represents white, 1.0 represents black.
	 *
	 */
	float * getOutput() {
		return float_frame;
	}

    cv::Mat getOutput_c1() {
        //return greyscaled_frame;
        return original_frame;
        //return flipped_frame;
    }

    cv::Mat & getOutputColorMat(){
        return flipped_frame;
    }

	/**
	 * Shows the output of the video or webcam to the screen in a window
	 */
	//see http://opencv.willowgarage.com/documentation/cpp/user_interface.html#cv-namedwindow
    void enableDisplayWindow(string win_title) {
        cv::namedWindow(win_title, CV_WINDOW_AUTOSIZE);
        this->win_title = win_title;
        showWindow = true;
    }

    void enableDisplayWindow() {
        enableDisplayWindow(win_title);
    }

	/**
	 * When set to true, "Canny" edge detection is applied to the video source
	 * ( shape outlines are shown in video)
	 */
	void setDoesEdgeDetection(bool on_off){
		edge_detection = on_off;
	}
    /**
	 * grab - grabs a frame from the video source.
	 * Returns true if it could retrieve one, otherwize returns false
	 *
	 * Ment to be used in a while loop to keep capturing until the end of the video.
	 */
	bool grab();

    /** rewinds the video
     *  @return - true if it did rewind false otherwise
     */
    bool rewind();

    // 2013.6.25
    void splitBGR()
    {
        int channels = this->flipped_frame.channels();
        int height = this->flipped_frame.rows;
        int width = this->flipped_frame.cols;

        std::vector<cv::Mat> bgr(channels);
        cv::split(this->flipped_frame, bgr);

        convert(bgr[0], bFrame);
        convert(bgr[1], gFrame);
        convert(bgr[2], rFrame);

//#define SHOW_SPLIT
#ifdef SHOW_SPLIT
        // Create a black board
        cv::Mat bk;
        bk.create(this->flipped_frame.rows, this->flipped_frame.cols, CV_8UC1);
        bk = cv::Scalar(0);
        // Used for merging
        std::vector<cv::Mat> mbgr(channels);
        cv::Mat bm;
        cv::Mat gm;
        cv::Mat rm;
        // B
        mbgr[0] = bgr[0];
        mbgr[1] = bk;
        mbgr[2] = bk;
        cv::merge(mbgr, bm);
        // G
        mbgr[0] = bk;
        mbgr[1] = bgr[1];
        mbgr[2] = bk;
        cv::merge(mbgr, gm);
        // R
        mbgr[0] = bk;
        mbgr[1] = bk;
        mbgr[2] = bgr[2];
        cv::merge(mbgr, rm);
        //
        cv::imshow("bgr", this->flipped_frame);
        cv::imshow("b", bm);
        cv::imshow("g", gm);
        cv::imshow("r", rm);
#endif
    }

    // 2013.6.25
    float * getBFrame()
    {
        return bFrame;
    }
    float * getGFrame()
    {
        return gFrame;
    }
    float * getRFrame()
    {
        return rFrame;
    }
    void turnOnColor()
    {
        if(!isShowColor){
            isShowColor = true;
            bFrame = new float[target_size.area()];
            gFrame = new float[target_size.area()];
            rFrame = new float[target_size.area()];
        }
    }
};

#endif /* VIDEOSOURCE_H_ */
