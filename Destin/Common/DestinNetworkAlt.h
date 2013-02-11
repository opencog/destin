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
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "INetwork.h"

extern "C" {
#include "destin.h"
#include "node.h"
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
    W512 = 512  //8 128x128 layers needed
};
#define MAX_IMAGE_WIDTH 512

using std::string;

class DestinNetworkAlt: public INetwork {

private:
    Destin * destin;
    bool training;
    DestinIterationFinishedCallback * callback;

    float beta;   //variance learning rate
    float lambda; //previous belief damping, 0 = disable, 1.0 = keep the same
    float gamma;  //parents previous belief damping, 0 = disable, 1.0 = keep the same

    float * temperatures; //one temperature per layer. If temperature = #
                          //centroids then the belief distribution doesn't change.
                          //Higher temperature makes the belief more winner
                          //take all, lower temperature makes it more uniform.

    float centroidImageWeightParameter; // higher value gives centroids images higher contrast

    cv::Mat winningGrid;
    cv::Mat winningGridLarge;
    cv::Mat centroidImage;
    cv::Mat centroidImageResized;
    cv::Mat layerCentroidsImage;//one large image of all centroid images of one layer put together in a grid fashion

    float *** centroidImages;
    bool isUniform;
    /**
     * initTemperatures
     * Input layer stays the same. For the top two layers temperature = centroids * 2
     * Because the input layer is generally already sparse and the top layers
     * are more uniform and need help to becoming more decisive.
     */
    void initTemperatures(int layers, uint * centroids);

    float *** getCentroidImages();

    void decodeLabelLayerCentroid(short label, int & layer_out, int & centroid_out);

    int decodeLabelChildNum(short label);
    void paintCentroidImage(int cent_layer, int centroid, int x, int y, cv::Mat & img);

    void calcChildCoords(int px, int py, int child_no, int child_layer, int & child_x_out, int & child_y_out);

    void displayTreeHelper(std::vector<short> & tree, int tree_pos, int px, int py, cv::Mat & img);
public:

    DestinNetworkAlt(SupportedImageWidths width, unsigned int layers,
            unsigned int centroid_counts [], bool isUniform );

    virtual ~DestinNetworkAlt();

    void setBeliefTransform(BeliefTransformEnum e){
        SetBeliefTransform(destin, e);
    }

    void setTemperatures(float temperatures[]);

    void doDestin( //run destin with the given input
            float * input_dev //pointer to input memory on device
            );

    void isTraining(bool isTraining);

    bool isTraining() {
        return training;
    }

    void free() {
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

    int getBeliefsPerNode(int layer){
        return destin->nb[layer];
    }

    float * getNodeBeliefs(int layer, int row, int col){
        return getNode(layer, row, col)->pBelief;
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

    void displayFeatures(int layer, int node_start, int nodes){
        DisplayLayerFeatures(destin, layer, node_start, nodes);
    }

    /** print given node's centroid's locations
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
      * by the node's winning_centroid_index / # of centroids.
      * @param layer = layer to show
      * @param zoom = scales the image by this factor
      */
    void imageWinningCentroidGrid(int layer, int zoom = 8, const string& window_name="Winning Grid");

    void printNodeObservation(int layer, int row, int col);

    void printNodeBeliefs(int layer, int row, int col);

    void setParentBeliefDamping(float gamma);

    void setPreviousBeliefDamping(float lambda);

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
     * Destroy the current destin structure before loading the new one.
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


    void updateCentroidImages(){
        Cig_UpdateCentroidImages(destin, getCentroidImages(), centroidImageWeightParameter);
    }

    /** Returns the centroid image as a float array.
      * Then displays it.
      */
    float * getCentroidImage(int layer, int centroid);

    /** Gets the generated centroid images as an opencv Mat image.
      */
    cv::Mat getCentroidImageM(int layer, int centroid, int disp_width = 256, bool enhanceContrast = false);

    /** Generates representative images for all centroids then displays the chosen image.
      * Currently only works for uniform destin. cvWaitKey() must be called after to see the image.
      * The contrast of the image can be enhanced by setting the weight exponent
      * parameter larger than 1.0 via setCentImgWeightExponent() method, and further by
      * passing enhanceContrast parameter as true.
      * @param layer - which layer the centroid belongs
      * @param centroid - which centroid of the given layer to display.
      * @param disp_width - scales the square centroid image to this width to be displayed. Defaults to 256 pixels
      * @param enhanceContrast - if true, the image contrast is enhanced using the opencv function cvEqualizeHist as a post processing step
      * @param window_name - name to give the display window
      */
    void displayCentroidImage(int layer, int centroid, int disp_width = 256, bool enhanceContrast = false, string window_name="Centroid Image" );

    void saveCentroidImage(int layer, int centroid, string filename, int disp_width = 256, bool enhanceContrast = false);

    /** Displays all the centroid images of a layer into one large image.
      * The centroid images are arranged into a grid and are seperated by a black boarder.
      * If there's not a square number of centroid images, then empty (black) centroid images
      * are added to the bottom right of the grid until the number is a square number. For example
      * if there are 15 centroid images, then it will display a 4x4 grid with the bottom right corner empty.
      * Only works with uniform destin.
      * cvWaitKey() must be called after for the image to appear.
      *
      * @param layer - what layer the centroids belong to
      * @param scale_width - resizes the square image grid width to this size.
      * @param boarder_width - how wide, in pixels, the black border is that seperates the images.
      * @param window_title - name given to the window that is displayed.
      */
    void displayLayerCentroidImages(int layer,
                                    int scale_width = 600,
                                    int border_width = 5,
                                    string window_title="Centroid Images"
                                    );


    /** Displays an image representation of the tree.
      * The input tree descibes
      */
    void displayTree(std::vector<short> & tree);

};


#endif /* DESTINNETWORKALT_H_ */
