#ifndef DST_IMAGE_SOURCE_BASE_H
#define DST_IMAGE_SOURCE_BASE_H


#include <vector>
#include <opencv/cv.h>

using namespace std;
class ImageSourceBase {

protected:
    int currentImage;               // index into images vector
    std::vector<cv::Mat> grayMats;  // store images as grayscal opencv Mats (matrix, images)
    int nImages;                    // number of images in the batch

    std::vector<cv::Mat> colorMats;  // store images as color opencv Mats (matrix, images)


    virtual bool isImageIncluded(int index) = 0;
public:

    ImageSourceBase(int nImages)
        :currentImage(-1),
          nImages(nImages){
    }

    ImageSourceBase()
        :currentImage(-1),
          nImages(0){
    }

    /** Number of images added or available
     */
    int getImageCount(){
        return nImages;
    }

    /**
      * @return - integer of the "current image" id.
      */
    int getImageIndex(){
        return currentImage;
    }
    /** Sets the current image.
      * Methods such as getGrayImageFloat will reflect what this is set to.
      * The next invocation of the findNextImage method will start from this location.
      * @param id - image id between 0 and nImages
      */
    void setCurrentImage(uint image_id){
        if(image_id > nImages){
            throw std::domain_error("setCurrentImage: id out of range\n");
        }
        currentImage = image_id;
    }

    /** Gets the current image represented as a float vector.
      * Suitable to be fed into DeSTIN. The grayscale pixels are
      * floats from 0.0 to 1.0;
      */
    float * getGrayImageFloat(){
        return (float *)grayMats[currentImage].data;
    }

    // 2013.5.7
    // CZT
    //
    float * getGrayImageFloat_c1()
    {
        return (float *)getGrayImageMat(512, 512).data;
    }

    /** Same as getGrayImageFloat()
      * Here to be compatible with VideoSource
      */
    float * getOutput(){
        return getGrayImageFloat();
    }

    /** Gets the current image as an opencv color Mat (image.)
      * @params rows, cols - scales the image to this size
      */
    cv::Mat getColorImageMat(int rows = 32, int cols = 32){
        if(rows != 32 || cols != 32){
            cv::Mat image = colorMats[currentImage];
            cv::Mat bigger;
            cv::resize(image, bigger, cv::Size(rows, cols), 0, 0, CV_INTER_NN);
            return bigger;
        }else{
            return colorMats[currentImage];
        }
    }



    /** Displays the given image to the user in a window.
      * A call to cv::waitKey() must be called by the user for it to show.
      * @param image_id - The image to show between 0 and 9999.
      *                   The image_id can be different from and does not
      *                   change the "current image".
      */
    void displayCifarColorImage(int image_id, int rows=512, int cols=512, string window_title="CIFAR Color Image"){
        if(image_id < 0 || image_id >= nImages){
            printf("displayCifarColorImage, index out of bounds\n");
            return;
        }

        cv::Mat image = colorMats[image_id];
        cv::Mat bigger;
        cv::resize(image, bigger, cv::Size(rows, cols), 0, 0, CV_INTER_NN);
        cv::imshow(window_title, bigger);
    }

    /** Same as displayCifarGrayImage method, except in grayscale
      */
    void displayCifarGrayImage(int index, int rows=512, int cols=512, string window_title="CIFAR Gray Image"){
        if(index < 0 || index >= nImages){
            printf("displayCifarGrayImage, index out of bounds\n");
            return;
        }

        cv::Mat bigger;
        cv::resize(grayMats[index], bigger, cv::Size(rows, cols), 0, 0, CV_INTER_NN);
        cv::imshow(window_title, bigger);
    }

    /** Returns the "current image" as an opencv image (Mat)
      * Not suitable for input to DeSTIN, use getGrayImageFloat method instead.
      * @param rows, cols - optionally scale the image to this size
      */
    cv::Mat getGrayImageMat(int rows = 32, int cols = 32){
        if(rows != 32 || cols != 32){
            cv::Mat image = grayMats[currentImage];
            cv::Mat bigger;
            cv::resize(image, bigger, cv::Size(rows, cols), 0, 0, CV_INTER_NN);
            return bigger;
        }else{
            return grayMats[currentImage];
        }
    }

    bool grab(){
        findNextImage();
        return true;
    }

    /** Finds the next image to show.
      * Finds an image that has a class that is enabled.
      * If it finds an image, it remebers the spot so another call
      * to this method will start from that spot.
      * Wraps to the begining of the batch when it reaches the end.
      * @throws logic_error - if it cant find an image
      * @returns index of the image found
      */
    int findNextImage(){
        //start searching from current location
        for(int i = currentImage + 1; i < nImages; i++){
            if( isImageIncluded(i)){
                currentImage = i;
                return currentImage;
            }
        }

        //not found, so start from the begining
        for(int i = 0; i < nImages; i++){
            if( isImageIncluded(i)){
                currentImage = i;
                return currentImage;
            }
        }

        //still not found
        currentImage = -1;
        throw std::logic_error("findNextImage: could not find an image\n.");

    }
};

#endif
