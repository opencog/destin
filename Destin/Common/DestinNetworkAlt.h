/*
 * destinnetworkalt.h
 *
 *  Created on: Jun 23, 2012
 *      Author: ted
 */

#ifndef DESTINNETWORKALT_H_
#define DESTINNETWORKALT_H_

#include <stdexcept>
#include <vector>
#include <iostream>
#include <math.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "INetwork.h"

extern "C" {
#include "destin.h"
#include "node.h"
#include "centroid.h"
#include "cent_image_gen.h"
}

#include "DestinIterationFinishedCallback.h"

enum SupportedImageWidths {
    W4 = 4,     //1 1x1
    W8 = 8,     //2 2x2
    W16 = 16,   //3 4x4
    W32 = 32,   //4 8x8
    W64 = 64,   //5 16x16
    W128 = 128, //6 32x32
    W256 = 256, //7 64x64
    W512 = 512  //8 layers 128x128 nodes
};

enum DstImageMode {
    DST_IMG_MODE_GRAYSCALE,
    DST_IMG_MODE_RGB,
    DST_IMG_MODE_GRAYSCALE_STEREO
};

using std::string;

class DestinNetworkAlt: public INetwork {

private:
    Destin * destin;
    bool training;
    DestinIterationFinishedCallback * callback;

    float lambdaCoeff; //previous belief damping, 0 = disable, 1.0 = keep the same
    float gamma;  //parents previous belief damping, 0 = disable, 1.0 = keep the same

    //One temperature per layer. If temperature = #
    //centroids then the belief distribution doesn't change.
    //Higher temperature makes the belief more winner
    //take all, lower temperature makes it more uniform.
    float * temperatures;

    float centroidImageWeightParameter; // higher value gives centroids images higher contrast

    cv::Mat winningGrid;
    cv::Mat winningGridLarge;
    cv::Mat centroidImageResized;
    cv::Mat layerCentroidsImage;//one large image of all centroid images of one layer put together in a grid fashion

    float **** centroidImages;
    bool isUniform;
    const int inputImageWidth;

    DstImageMode imageMode; //Grayscale or color input

    /**
     * initTemperatures
     * Input layer stays the same. For the top two layers temperature = centroids * 2
     * Because the input layer is generally already sparse and the top layers
     * are more uniform and need help to becoming more decisive.
     */
    void initTemperatures(int layers, uint * centroids);

    void init(SupportedImageWidths width, unsigned int layers,
              unsigned int centroid_counts [], bool isUniform,
              int extRatio,  unsigned int layer_widths[]);

    /** Convert raw float array centroid image to OpenCV mat.
      * @param toByteType - if true, then convert to a CV_U8 byte type, otherwise
      * it will be a CV_32F type
      */
    cv::Mat convertCentroidImageToMatImage(int layer, int centroid, bool toByteType);

    //get OpenCV float image/mat type depending on the imageMode
    int getCvFloatImageType();
    //get OpenCV byte image/mat type depending on the imageMode
    int getCvByteImageType();

    /** Get destin extRatio for the given image mode
     * grayscale = 1, RGB = 3
     */
    int getExtendRatio(DstImageMode imageMode);
    DstImageMode extRatioToImageMode(int extRatio);

    /*** helper methods for rescaling centroids ***/
    void getSelectedCentroid(int layer, int idx, std::vector<float> & outCen);
    void normalizeChildrenPart(std::vector<float> & inCen, int ni);
    // Prints all centroids
    void printFloatCentroids(int layer);
    // Prints the vector with a given title
    void printFloatVector(std::string title, std::vector<float> vec);
    void rescaleRecursiveUp(int srcLayer, std::vector<float> selCen, int dstLayer);
    void rescaleRecursiveDown(int srcLayer, std::vector<float> selCen, int dstLayer);

public:

    /** Builds a DeSTIN network
      * @param width - width of the input image in pixels
      * @param layers - number of layers in the heirarchy
      * @param centroid_counts - centroids per layer. Starts from the bottom layer.
      * @param isUniform - if nodes in a level share the same pool of centroids.
      * @param layer_widths - if null, it will build a classic 4 to 1 non overlapping heirarchy based on the
      * number of layers. Otherwise it will build the heirarchy according to these rules:
      * 1) If parent layer width is exactly one less than the child layer width, it assumes the parent nodes share their
      * children with other adjacent parents in an overlapping fashion. Each child may have up to 4 parents. Chilren nodes
      * on the edges and corners of the child layer will have less than 4 parents.
      * 2) Otherwise, if child_layer_width % parent_layer_width == 0 then it assumes that parent nodes do not share children nodes
      * Each child has just 1 parent. Each parent will have (child_layer_width / parent_layer_width) ^ 2 children.
      * 3) If neither of the two conditions do not apply, a runtime_error exception will be thrown.
      * @param imageMode - specify wheter using grayscale or color images
      */
    DestinNetworkAlt(SupportedImageWidths width, unsigned int layers,
                     unsigned int centroid_counts [], bool isUniform,
                     unsigned int layer_widths[] = NULL,
                     DstImageMode imageMode = DST_IMG_MODE_GRAYSCALE);

    virtual ~DestinNetworkAlt();

    /** Runs the DeSTIN algorithm on the float array input.
      * This is usually a square greyscale image made of float values between 0.0 and 1.0
      * The size of this array should be equal width x width, where width was passed
      * into the DestinNetworkAlt constructor.
      */
    void doDestin(float * input_array);

    int getInputImageWidth(){
        return inputImageWidth;
    }

    /** Gets centroid images as raw float arrays
      * To get a particular centroid image:
      * float * image_array = getCentroidImages()[channel][layer][centroid];
      * The size of the image_array be determined from:
      * width = Cig_GetCentroidImageWidth(destin, layer); size = width * width;
      * The number of channels is destin->extRatio, which would be
      * 1 channel for grayscale and 3 for RGB images.
      */
    float**** getCentroidImages();

    void setBeliefTransform(BeliefTransformEnum e){
        SetBeliefTransform(destin, e);
    }

    void setTemperatures(float temperatures[]);

    void setFrequencyCoefficients(float freqCoeff, float freqTreshold, float addCoeff);
    void setStarvationCoefficient(float starvCoeff);
    void setMaximumCentroidCounts(int count);

    float getSep(int layer);
    float getVar(int layer);
    float getQuality(int layer);

    std::vector<float> getLayersVariances();
    std::vector<float> getLayersSeparations();
    std::vector<float> getLayersQualities();

    /*************************************************************************/
    void rescaleCentroid(int srcLayer, int idx, int dstLayer);


    void isTraining(bool isTraining);

    bool isTraining() {
        return training;
    }

    void destroy() {
        DestroyDestin(destin);
        destin = NULL;
    }

    void setFixedLearnRate(float rate){
        destin->fixedLearnRate = rate;
    }

    void setIsPOSTraining(bool training) {
        isTraining(training);
    }

    void setIsPSSATraining(bool no_op) {
        //this network doesn't do pssa training
        std::cout << "DestinNetworkAlt setIsPSSATraining called. Currently a noop.\n";
    }

    void setIterationFinishedCallback(DestinIterationFinishedCallback * callback){
        this->callback = callback;
    }

    /**
     * @return - Number of centroids or beliefs per node.
     */
    int getBeliefsPerNode(int layer){
        return destin->nb[layer];
    }

    float * getNodeBeliefs(int layer, int row, int col){
        return getNode(layer, row, col)->belief;
    }

    /**
     * Fills beliefs array in belief values for all nodes from given layer.
     * The caller should allocate the array.
     */
    void getLayerBeliefs(int layer, float * beliefs){
        GetLayerBeliefs(destin, layer, beliefs);
    }

    Node * getNode(int layer, int row, int col){
        Node * n = GetNodeFromDestin(destin, layer, row, col);
        if(n == NULL){
            throw std::logic_error("could't getNode");
        }
        return n;
    }

    Destin * getNetwork(){
        return destin;
    }

    /** Print given node's centroid's locations.
     *  The Centroids consists of 3 main parts, the dimensions which cluster
     *  on the nodes input, the dimensions which cluster on its previous belief
     *  and the dimensions which cluster on the parents previous belief.
     */
    void printNodeCentroidPositions(int layer, int row, int col);

    /** Prints a grid of winning centroid numbers for a layer.
      * The grid is the size of the layer in nodes. Each
      * node has a winning centroid number.
      */
    void printWinningCentroidGrid(int layer);

    /** Shows an image representing winning centroids
      * Each pixel represents a node where the grayscale color is determined
      * by the node's winning_centroid_index divided by # of centroids per node.
      * @param layer = layer to show
      * @param zoom = scales the image by this factor
      */
    void imageWinningCentroidGrid(int layer, int zoom = 8, const string& window_name="Winning Grid");

    void printNodeObservation(int layer, int row, int col);

    void printNodeBeliefs(int layer, int row, int col);

    void setParentBeliefDamping(float gamma);

    void setPreviousBeliefDamping(float lambdaCoeff);

    void clearBeliefs(){
        ClearBeliefs(destin);
    }

    void setLayerIsTraining(uint layer, bool isTraining){
        destin->layerMask[layer] = isTraining;
    }

    unsigned int getLayerCount(){
        return destin->nLayers;
    }

    /** Saves the current destin network to file
     * Includes centroid locations, and current and previous beliefs.
     */
    void save(char * fileName){
        SaveDestin(destin, fileName);
    }

    /** Loads a destin structure from the given file
     * Destroys the current destin structure before loading the new one.
     */
    void load(const char * fileName);

    /** Moves the given centroid to its node's input last observation.
      * @param layer
      * @param row
      * @param col
      * @param centroid - centroid to move
      */
    void moveCentroidToInput(int layer, int row, int col, int centroid);

    /** Helps determine the centroid image contrast.
      * Values greater than 1.0 will provide more contrasts to
      * the centroid images generated via displayCentroidImage() method.
      */
    void setCentImgWeightExponent(float exp){
        centroidImageWeightParameter = exp;
    }

    /** Updates the generated centroid images for all layers.
      */
    void updateCentroidImages(){
        Cig_UpdateCentroidImages(destin, getCentroidImages(), centroidImageWeightParameter);
    }

    /** Returns the centroid image as a float array.
      * Then displays it.
      * Opencv cvWaitKey function must be called after to see the image.
      */
    float * getCentroidImage(int channel, int layer, int centroid);

    /** Returns generated centroid image.
      * Method updateCentroidImages should be called to get a refreshed version of the image.
      * Currently only works when isUniform instance variable is true.
      * The contrast of the image can be enhanced by setting the weight exponent
      * parameter larger than 1.0 via setCentImgWeightExponent method, and further by
      * passing enhanceContrast parameter as true.
      * @param layer - which layer the centroid belongs
      * @param centroid - which centroid of the given layer to display.
      * @param disp_width - scales the square centroid image to this width to be displayed.
      * Defaults to 256 pixels
      * @param enhanceContrast - if true, the image contrast is enhanced using the opencv function
      * cvEqualizeHist as a post processing step
      * @param window_name - name to give the display window
      */
    cv::Mat getCentroidImageM(int layer, int centroid, int disp_width = 256,
                              bool enhanceContrast = false);


    /** Displays the image generated from getCentroidImageM method.
      * updateCentroidImages() should be called to recalulate the images to see changes.
      * Opencv cvWaitKey function must be called after to see the image.
      */
    void displayCentroidImage(int layer, int centroid, int disp_width = 256,
                              bool enhanceContrast = false, string window_name="Centroid Image" );

    /** Saves the image generated from getCentroidImageM method.
      */
    void saveCentroidImage(int layer, int centroid, string filename, int disp_width = 256,
                           bool enhanceContrast = false);

    /** Displays all the centroid images of a layer into one large image.
      * Displays the image generated from method getLayerCentroidImages.
      * updateCentroidImages() should be called to recalulate the images to see changes.
      * Opencv cvWaitKey function must be called after to see the image.
      * @param layer - what layer the centroids belong to
      * @param scale_width - resizes the square image grid width to this size.
      * @param boarder_width - how wide, in pixels, the black border is that seperates the images.
      * @param window_title - name given to the window that is displayed.
      * @param sorted_centroids - if true, will try to arrange the centroid images according to appearance
      */
    void displayLayerCentroidImages(int layer,
                                    int scale_width = 600,
                                    int border_width = 5,
                                    string window_title="Centroid Images",
                                    std::vector<int> sort_order = std::vector<int>());


    /** Creates an image that arranges all the centroid images of a layer into a grid.
      * The centroid images are arranged into a grid and are seperated by a black boarder.
      * If there's not a square number of centroid images, then empty (black) centroid images
      * are added to the bottom right of the grid until the number is a square number. For example
      * if there are 15 centroid images, then it will display a 4x4 grid with the bottom right corner empty.
      * Only works with uniform destin.
      * updateCentroidImages() should be called to recalulate the images to see changes.
      *
      * @param layer - what layer the centroids belong to
      * @param scale_width - resizes the square image grid width to this size.
      * @param boarder_width - how wide, in pixels, the black border is that seperates the images.
      * @param sorted_centroids - if true, will try to arrange the centroid images according to appearance
      */
    cv::Mat getLayerCentroidImages(int layer,
                                   int scale_width = 1000,
                                   int border_width = 5, std::vector<int> sort_order=std::vector<int>());

    /** Saves the image generated from getLayerCentroidImages method to disk.
      * @param layer - which layer to generate the image for
      * @param filename - Where to save the file. The image format is based on the file extention.
      *     Has been tested on *.png.
      * @param scale_width - resizes the square image grid width to this size.
      * @param border_width -  how wide, in pixels, the black border is that seperates the images.
      * @param sorted_centroids - if true, will try to arrange the centroid images according to appearance
      */
    void saveLayerCentroidImages(int layer, const string & filename,
                                 int scale_width = 1000,
                                 int border_width = 5,
                                 std::vector<int> sort_order = std::vector<int>());


    /**
     * Calculate rectilinear (taxicab) distance between two centroid points
     */
    float distanceBetweenCentroids(int layer, int centroid1, int centroid2);

    /**
     * Returns the indicies of centroids for a layer sorted according to apearance.
     * This tries to group similar looking centroids to be near each other.
     * The returned list can be passed to saveLayerCentroidImages and getLayerCentroidImages methods
     * to have them show the images in the given order.
     *
     * @param layer
     * @return - list of centroid incies in the sorted order
     */
    std::vector<int> sortLayerCentroids(int layer);

};


#endif /* DESTINNETWORKALT_H_ */
