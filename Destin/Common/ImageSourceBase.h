#ifndef DST_IMAGE_SOURCE_BASE_H
#define DST_IMAGE_SOURCE_BASE_H


#include <vector>
#include <opencv/cv.h>
#include <macros.h>

using namespace std;
class ImageSourceBase {

private:
    const int rows, cols; // default output size of the images
    const int imageSize; // rows * cols

    cv::Mat resizeImage(cv::Mat & image, int rows, int cols){
        rows = rows == -1 ? this->rows : rows;
        cols = cols == -1 ? this->cols : cols;

        if(rows != image.rows || cols != image.cols ){
            cv::Mat bigger;
            cv::resize(image, bigger, cv::Size(rows, cols), 0, 0, CV_INTER_NN);
            return bigger;
        }else{
            return image;
        }
    }

protected:

    int currentImage;               // index into images vector
    std::vector<cv::Mat> grayMats;  // store images as grayscal opencv Mats (matrix, images)
    int nImages;                    // number of images in the batch

    std::vector<cv::Mat> colorMats;  // store images as color opencv Mats (matrix, images)

    /** Return true if the given image is allowed to be shown.
      * Subclasses should override to determine if the image is allowed to be shown.
      */
    virtual bool isImageIncluded(int index) = 0;

    std::vector<float *> colorFloatImages; // store images as color RGB (not BRG) float arrays to be used by DeSTIN

    void convertMatToFloat(cv::Mat & colorMat, float * colorImageBuff_out){

        if(colorMat.type() != CV_8UC3){
            std::runtime_error("getBGRImageFloat(): had unexpected matrix type.\n");
        }

        cv::Point p(0, 0);
        int i = 0 ;
        for (p.y = 0; p.y < colorMat.rows; p.y++) {
            for (p.x = 0; p.x < colorMat.cols; p.x++) {
                colorImageBuff_out[i + imageSize * 0] = (float)(colorMat.at<cv::Vec3b>(p)[0]) / 255.0; // blue
                colorImageBuff_out[i + imageSize * 1] = (float)(colorMat.at<cv::Vec3b>(p)[1]) / 255.0; // green
                colorImageBuff_out[i + imageSize * 2] = (float)(colorMat.at<cv::Vec3b>(p)[2]) / 255.0; // red
                i++;
            }
        }
    }


    /** Create a float color RGB image to be used by DeSTIN and save it for later
    * to be used by getBGRImageFloat()
    */
    void addColorFloatImage(cv::Mat & colorIm){
        cv::Mat mixedMat(colorIm.rows, colorIm.cols, colorIm.type());
        cv::cvtColor(colorIm, mixedMat, CV_BGR2RGB);  // switch from BGR to RGB
        float * colorFloatImage = new float[imageSize * 3];
        convertMatToFloat(mixedMat, colorFloatImage);
        colorFloatImages.push_back(colorFloatImage);
    }

public:

    /** Constructor
      *
      */
    ImageSourceBase(int rows, int cols)
        :currentImage(-1), nImages(0), rows(rows), cols(cols),
          imageSize(rows * cols){ }

    virtual ~ImageSourceBase(){
        for(int i = 0 ; i < colorFloatImages.size() ; i++){
            delete colorFloatImages[i];
        }
        colorFloatImages.clear();
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
      * floats from 0.0 to 1.0.
      * User should not delete the returned array.
      */
    float * getGrayImageFloat(){
        return (float *)grayMats[currentImage].data;
    }

    /** Get the current color image as a float vector.
      * This is suitable to fed into DeSTIN if the extendRatio == 3 ( see DestinNetworkAlt constructor ).
      *
      * The three color component float images (R, G, B) are append one after the other.
      * The length of the array is 3 * rows * cols. Rows and cols is the image size
      * as specified in the constructor.
      * The float values are between 0.0 and 1.0
      *
      * User should not delete the returned array.
      * @return - pointer to color image float array to be fed into DeSTIN
      */
    float * getRGBImageFloat(){
        return colorFloatImages.at(currentImage);
    }

    /** Same as getGrayImageFloat()
      * Here to be compatible with VideoSource
      */
    float * getOutput(){
        return getGrayImageFloat();
    }

    /** Gets the current image as an opencv color Mat (image.)
      * @params rows, cols - scales the image to this size
      * If not specifed, it defaults to the rows, cols as passed into the constructor
      */
    cv::Mat getColorImageMat(int rows = -1, int cols = -1){
        return resizeImage(colorMats[currentImage], rows, cols);
    }

    /** Returns the "current image" as an opencv image (Mat)
      * Not suitable for input to DeSTIN, use getGrayImageFloat method instead.
      * @param rows, cols - optionally scale the image to this size
      * If not specifed, it defaults to the rows, cols as passed into the constructor
      */
    cv::Mat getGrayImageMat(int rows = -1, int cols = -1){
        return resizeImage(grayMats[currentImage], rows, cols);
    }

    /** Displays the given image to the user in a window.
      * A call to cv::waitKey() must be called by the user for it to show.
      * @param image_id - The image to show between 0 and 9999.
      *                   The image_id can be different from and does not
      *                   change the "current image".
      * @param rows, cols - optionally scale the image to this size
      * If not specifed, it defaults to the rows, cols as passed into the constructor
      */
    void displayCifarColorImage(int image_id, int rows=-1, int cols=-1, string window_title="CIFAR Color Image"){
        if(image_id < 0 || image_id >= nImages){
            printf("displayCifarColorImage, index out of bounds\n");
            return;
        }

        cv::Mat bigger = resizeImage(colorMats[image_id], rows, cols);
        cv::imshow(window_title, bigger);
    }

    /** Same as displayCifarColorImage method, except in grayscale
      */
    void displayCifarGrayImage(int image_id, int rows=-1, int cols=-1, string window_title="CIFAR Gray Image"){
        if(image_id < 0 || image_id >= nImages){
            printf("displayCifarGrayImage, index out of bounds\n");
            return;
        }

        cv::Mat bigger = resizeImage(grayMats[image_id], rows, cols);
        cv::imshow(window_title, bigger);
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
