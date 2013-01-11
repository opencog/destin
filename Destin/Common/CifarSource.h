#ifndef CIFAR_SOURCE_H
#define CIFAR_SOURCE_H


#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include <map>

#include <opencv/cv.h>
#include <opencv/highgui.h>


#include "ImageSourceBase.h"

using namespace std;


typedef unsigned int uint;

class CifarSource : public ImageSourceBase {
    /** see http://www.cs.toronto.edu/~kriz/cifar.html
      for info about CIFAR images.
      */

    string cifar_dir;

    const int batch;            // batch being used Immutable

    const int numClasses;       // Number of classes, i.e. airplane, dog... currently = 10

    bool * classesEnabled;      // one boolean for each class, true for enabled otherwise disabled

    map< string, unsigned int> name_to_class; // maps the class name to the class number

    const int       imageSize;  // size of an image in bytes ( label + image data)

    const size_t    batch_size; // size of raw batch data in bytes
    uchar *         raw_batch;  // raw batch file data

    typedef struct {
        int classLabel;         // image class indicated by an integer
        size_t image_offset;    // index in raw_batch data array
        uchar * image;          // pointer to the image data
    } image_info;

    vector<image_info> images;  // saves info about all the images


    void setupNameToClass(){
        name_to_class["airplane"] = 0;
        name_to_class["automobile"] = 1;
        name_to_class["bird"] = 2;
        name_to_class["cat"] = 3;
        name_to_class["deer"] = 4;
        name_to_class["dog"] = 5;
        name_to_class["frog"] = 6;
        name_to_class["horse"] = 7;
        name_to_class["ship"] = 8;
        name_to_class["truck"] = 9;
    }

    cv::Mat convertToColorMat(uchar * image){
        cv::Mat mat(32, 32, CV_8UC3);

        uchar * data = mat.data;

        int k = 0;
        for(int i = 0 ; i < 32 * 32 ; i++){
            //opencv actually uses BGR instead of RGB
            data[k++] = image[2 * 32 * 32 + i]; //B
            data[k++] = image[1 * 32 * 32 + i]; //G
            data[k++] = image[0 * 32 * 32 + i]; //R
        }
        return mat;
    }

    /** Reads the batch image file and divides it.
      * Divides it into indivual images saving info
      * about each one into a image_info struct.
      */
    void loadBatch(){
        //create batch filename from cifar directory
        stringstream fn;
        fn << cifar_dir << "/data_batch_" << batch << ".bin";

        //open it for reading
        FILE * f = fopen(fn.str().c_str(), "r");
        if(f == NULL){
            throw runtime_error("could not open file at "+fn.str()+"\n");
        }

        size_t nread = fread(raw_batch, sizeof(uchar), batch_size, f);
        if(nread != batch_size){
            throw runtime_error("error while reading cifar batch " + fn.str() +"\n");
        }

        // see http://www.cs.toronto.edu/~kriz/cifar.html for structure of raw batch image data
        size_t imageOffset = 0;

        for(int i = 0 ; i < nImages ; i++){
            images[i].classLabel = raw_batch[imageOffset];
            images[i].image_offset = imageOffset + 1; //skip the class label
            images[i].image = &raw_batch[images[i].image_offset]; //store pointer to the image part

            //turn it into an opencv color image
            cv::Mat temp = convertToColorMat(images[i].image);

            //save it
            colorMats.push_back(temp);

            //turn the image grey scale
            cv::Mat gray;
            cvtColor(temp, gray, CV_RGB2GRAY);

            // make it have float values between 0 and 1
            // suitable for input into destin
            cv::Mat floatgray(32,32,CV_32FC1);
            float * data = (float *)floatgray.data;
            for(int j = 0 ; j < 32 * 32 ; j++){
                data[j] = (float)gray.data[j] / 255.0;
            }

            grayMats.push_back(floatgray);

            imageOffset += imageSize;
        }
        fclose(f);
    }
protected:
     bool isImageIncluded(int index){
        return classesEnabled[images[index].classLabel];
     }

public:

    /** Constructor
      *
      * @param cifar_dir - the directory which contains the
      * CIFAR data ffiles data_batch_1.bin  to data_batch_5.bin
      *
      * @param batch - Integer 1 to 5 of which data_batch_*.bin file to use
      * as the image source
      * @throws runtime_error if the data_batch bin file cannot be loaded.
      */
    CifarSource(string cifar_dir, int batch ) :
        ImageSourceBase(10000),
        batch(batch),
        numClasses(10),
        images(nImages),
        imageSize(1 + 32*32*3),
        batch_size( imageSize * nImages )
      // 10000 images in a batch. Each image is 1 byte
      // class label, then 32x32 pixels by 3 colors (RGB)
    {

        struct stat s;
        lstat(cifar_dir.c_str(), &s);

        if(!S_ISDIR(s.st_mode)){
            throw runtime_error(string("not a directory: ")+cifar_dir);

        }
        this->cifar_dir = cifar_dir;

        classesEnabled = new bool[numClasses];

        for(int i = 0 ; i < numClasses ; i++){
            classesEnabled[i] = true;
        }

        setupNameToClass();
        raw_batch = new uchar[batch_size];

        loadBatch();
    }

    ~CifarSource(){
        delete [] classesEnabled;
        delete [] raw_batch;
    }


    /** Sets all image classes to not be found.
      * At least one class should be re-enabled to avoid an exception from
      * the findNextImage method because it would not be able to find any images
      * otherwise.
      */
    void disableAllClasses(){
        for(int i = 0 ; i < numClasses ; i++){
            classesEnabled[i] = false;
        }
    }

    /** Turn on an image class to be found be class number.
      * @param classLabel - integer 0 to 9 corresponding to airplane, etc. see setupNameToClass method
      * @param enabled - if true then images of this class can be found otherwise they will be skipped
      */
    void setClassIsEnabled(unsigned int classLabel, bool enabled){
        if(classLabel >= 10){
            string message("setClassIsEnabled: classLabel must be less than ");
            message+=numClasses;
            message+="\n";
            throw  std::domain_error(message);
        }

        classesEnabled[classLabel] = enabled;
    }

    /** Turn on an image class to be found by class name.
      * @param className - name such as "airplane" or "dog" to enable
      * @param enabled - if true then images of this class can be found otherwise they will be skippeds
      */
    void setClassIsEnabledByName(string className, bool enabled){
        if(name_to_class.find(className) == name_to_class.end()){
            string mess = "setClassIsEnabledByName could not find class named "+className;
            throw std::domain_error(mess);
        }
        classesEnabled[name_to_class[className]] = enabled;
    }

    /** Returns the current image's class label.
      * For example if the current image is a dog then this
      * will return 5.
      */
    uint getImageClassLabel(){
        return images[currentImage].classLabel;
    }

};

#endif
