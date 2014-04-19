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

#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>


#include "ImageSourceBase.h"

using namespace std;


typedef unsigned int uint;


/** This class loads CIFAR images to be used by DeSTIN.
    See http://www.cs.toronto.edu/~kriz/cifar.html for info about CIFAR images.

  Here is an excerpt from the above website about the file format of the CIFAR data:

    The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin, as well as test_batch.bin.
    Each of these files is formatted as follows:
    <1 x label><3072 x pixel>
    ...
    <1 x label><3072 x pixel>

    In other words, the first byte is the label of the first image, which is a number in the range 0-9.
    The next 3072 bytes are the values of the pixels of the image.

    The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue.

    The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.

    Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows.
    Therefore each file should be exactly 30730000 bytes long.
  */
class CifarSource : public ImageSourceBase {

    string cifar_dir;

    const int numClasses;       // Number of classes, i.e. airplane, dog... currently = 10

    bool * classesEnabled;      // one boolean for each class, true for enabled otherwise disabled

    map< string, unsigned int> name_to_class; // maps the class name to the class number

    int imageDataSize;          // size of an image in bytes ( label + image data)

    size_t batch_size;          // size of raw batch data in bytes
    uchar * raw_batch;          // raw batch file data

    typedef struct {
        int classLabel;         // image class indicated by an integer
        size_t image_offset;    // index in raw_batch data array
        uchar * image;          // pointer to the image data
    } image_info;

    vector<image_info> images;  // saves info about all the images

    vector<int> batches;        // which CIFAR batches to use.


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
    void loadBatch(int batch, int images_per_batch){
        //create batch filename from cifar directory
        stringstream fn;
        fn << cifar_dir;
#ifdef _WIN32 // use the right path seperator depending on the platform
        fn << "\\" ;
#else
        fn << "/" ;
#endif
        if(batch == 6){
            fn << "test_batch.bin" ;
        } else {
            fn << "data_batch_" << batch << ".bin";
        }

        //open it for reading
        FILE * f = fopen(fn.str().c_str(), "rb");
        if(f == NULL){
            throw runtime_error("could not open file at "+fn.str()+"\n");
        }

        size_t nread = fread(raw_batch, sizeof(uchar), batch_size, f);
        if(nread != batch_size){
            throw runtime_error("error while reading cifar batch " + fn.str() +"\n");
        }

        // see http://www.cs.toronto.edu/~kriz/cifar.html for structure of raw batch image data
        size_t imageOffset = 0;

        for(int i = 0 ; i < images_per_batch ; i++){

            image_info info;

            info.classLabel = raw_batch[imageOffset]; // first byte is the class label 0 to 9
            info.image_offset = imageOffset + 1; //skip the class label to get the offset of the begining of the image data
            info.image = &raw_batch[info.image_offset]; //store pointer to the begining of the image part

            //turn it into an opencv color image
            cv::Mat temp = convertToColorMat(info.image);

            images.push_back(info);

            //save it
            colorMats.push_back(temp);
            addColorFloatImage(temp);

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

            imageOffset += imageDataSize;
        }
        fclose(f);
    }

    void initialize(string cifar_dir,  vector<int> & batches){
        imageDataSize = 1 + 32 * 32 * 3;

        const int images_per_batch = 10000; // each cifar batch file as 10,000 images.
        nImages = batches.size() * images_per_batch;
        images.reserve(nImages);
        batch_size = imageDataSize * images_per_batch;
        struct stat s;
        stat(cifar_dir.c_str(), &s);

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

        for(int i = 0 ; i < batches.size() ; i++){
            loadBatch(batches[i], images_per_batch);
        }
    }

protected:

    /** Return true if the given image is allowed to be shown.
      */
     bool isImageIncluded(int index){
        return classesEnabled[images[index].classLabel];
     }

public:

     /** Constructor
       *
       * @param cifar_dir - the directory which contains the
       * CIFAR data files data_batch_1.bin to data_batch_5.bin
       *
       * @param batch - Integer 1 to 5 of which data_batch_*.bin file to use
       * as the image source, or 6 to specify the test batch
       * @throws runtime_error if the data_batch bin file cannot be loaded.
       */
     CifarSource(string cifar_dir, int batch) :
         ImageSourceBase(32, 32), numClasses(10)
     {
        vector<int> batches;
        batches.push_back(batch);
        initialize(cifar_dir, batches);
     }

    /** Constructor
      *
      * @param cifar_dir - the directory which contains the
      * CIFAR data files data_batch_1.bin to data_batch_5.bin
      *
      * @param batches - list of integers 1 to 5 of which data_batch_*.bin file to use
      * as the image source, or 6 to specify the test batch
      * @throws runtime_error if the data_batch bin file cannot be loaded.
      */
    CifarSource(string cifar_dir, vector<int> batches) :
        ImageSourceBase(32, 32), numClasses(10)
      // 10000 images in a batch. Each image is 1 byte
      // class label, then 32x32 pixels by 3 colors (RGB)
    {
        initialize(cifar_dir, batches);
    }

    virtual ~CifarSource(){
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
        if(classLabel >= numClasses){
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
