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
    void initTemperatures(int layers, uint * centroids){
        temperatures = new float[layers];

        for(int l = 0 ; l < layers ; l++){
            temperatures[l] = centroids[l];
        }

        if(layers >= 2){
            temperatures[layers - 1] *= 2;
        }
        if(layers >=3){
            temperatures[layers - 2] *= 2;
        }
    }

public:

    DestinNetworkAlt(SupportedImageWidths width, unsigned int layers,
            unsigned int centroid_counts [], bool isUniform ) :
            training(true),
            beta(.01),
            lambda(.1),
            gamma(.1),
            isUniform(isUniform),
            centroidImages(NULL),
            centroidImageWeightParameter(1.0)
            {

        uint input_dimensionality = 16;
        uint c, l;
        callback = NULL;
        initTemperatures(layers, centroid_counts);
        temperatures = new float[layers];
        bool doesBoltzman = false;
        float starv_coef = 0.05;
        uint n_classes = 0;//doesn't look like its used
        uint num_movements = 0; //this class does not use movements

        //figure out how many layers are needed to support the given
        //image width.
        bool supported = false;
        for (c = 4, l = 1; c <= MAX_IMAGE_WIDTH ; c *= 2, l++) {
            if (c == width) {
                supported = true;
                break;
            }
        }
        if(!supported){
            throw std::logic_error("given image width is not supported.");
        }
        if (layers != l) {
            throw std::logic_error("Image width does not match the given number of layers.");
        }
        destin = InitDestin(
                input_dimensionality,
                layers,
                centroid_counts,
                n_classes,
                beta,
                lambda,
                gamma,
                temperatures,
                starv_coef,
                num_movements,
                isUniform,
                doesBoltzman
         );


        SetLearningStrat(destin, CLS_FIXED);
        ClearBeliefs(destin);
        destin->fixedLearnRate = 0.1;
        isTraining(true);
    }

    virtual ~DestinNetworkAlt() {
        if(centroidImages != NULL){
            Cig_DestroyCentroidImages(destin,  centroidImages);
        }

        if(destin!=NULL){
            DestroyDestin(destin);
            destin = NULL;
        }
        if(temperatures!=NULL){
            delete [] temperatures;
            temperatures = NULL;
        }
    }

    void doDestin( //run destin with the given input
            float * input_dev //pointer to input memory on device
            ) {
        FormulateBelief(destin, input_dev);

        if(this->callback != NULL){
            this->callback->callback(*this );
        }
    }

    void isTraining(bool isTraining) {
        this->training = isTraining;
        for(int l = 0 ; l < destin->nLayers ; l++){
           destin->layerMask[l] = isTraining ? 1 : 0;
        }
    }

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
    void printNodeCentroidPositions(int layer, int row, int col){
        Node * n = getNode(layer, row, col);
        for(int centroid  = 0 ; centroid < n->nb ; centroid++){
            printf("centroid %i: input ", centroid);
            int dimension;
            for(dimension = 0 ; dimension < n->ni ; dimension++){
                if(dimension % (n->ni / 4) == 0){
                    printf("\n");
                }
                printf("%.5f ", n->mu[centroid*n->ns + dimension]);
            }
            printf("\n");
            printf(" self ");
            for(int end = dimension + n->nb ; dimension < end ; dimension++){
                printf("%.5f ", n->mu[centroid*n->ns + dimension]);
            }
            printf(" parent ");
            for(int end = dimension + n->np ; dimension < end ; dimension++){
                printf("%.5f ", n->mu[centroid*n->ns + dimension]);
            }
            printf("\n");
        }
    }

    /** Prints a grid of winning centroid numbers for a layer.
      * The grid is the size of the layer in nodes. Each
      * node has a winning centroid number.
      */
    void printWinningCentroidGrid(int layer){
        uint width = (uint)sqrt(destin->layerSize[layer]);
        uint nidx = 0;
        Node * nodelist = GetNodeFromDestin(destin, layer, 0, 0);
        printf("Winning centroids grid - layer %i:\n", layer);
        for( int r  = 0 ; r < width ; r++){
            for(int c = 0 ; c < width ; c++ , nidx++){
                printf("%i ", nodelist[nidx].winner);
            }
            printf("\n");
        }
    }

    /** Shows an image representing winning centroids
      * Each pixel represents a node where the grayscale color is determined
      * by the node's winning_centroid_index / # of centroids.
      * @param layer = layer to show
      * @param zoom = scales the image by this factor
      */
    void imageWinningCentroidGrid(int layer, int zoom = 8, const string& window_name="Winning Grid"){
        uint width = (uint)sqrt(destin->layerSize[layer]);

        //initialize or create new grid if needed
        if(winningGrid.rows != width || winningGrid.cols != width){
            winningGrid = cv::Mat(width, width, CV_32FC1);
        }
        int nb = destin->nb[layer];
        float * data = (float *)winningGrid.data;
        for(int n = 0 ; n < destin->layerSize[layer] ; n++){
            data[n]  = (float)GetNodeFromDestinI(destin, layer, n)->winner / (float) nb;
        }
        cv::resize(winningGrid,winningGridLarge, cv::Size(), zoom, zoom, cv::INTER_NEAREST);
        cv::imshow(window_name, winningGridLarge);
    }

    void printNodeObservation(int layer, int row, int col){
        Node * n = getNode(layer, row, col);
        printf("Node Observation: layer %i, row %i, col: %i\n", layer, row, col);
        printf("input: ");
        int i = 0;
        for(int c = 0 ; c < n->ni ; c++ ){
            if(c % (n->ni / 4) == 0){
                printf("\n");
            }
            printf("%f ", n->observation[i]);
            i++;
        }
        printf("\nPrevious Belief: ");
        for(int c  = 0 ; c < n->nb ; c++ ){
            printf("%f ", n->observation[i]);
            i++;
        }
        printf("\nParent prev belief: ");
        for(int c  = 0 ; c < n->np ; c++ ){
            printf("%f ", n->observation[i]);
            i++;
        }
        printf("\n");
    }

    void printNodeBeliefs(int layer, int row, int col){
        Node * n = getNode(layer, row, col);
        printf("Node beliefs: layer%i, row %i, col: %i\n", layer, row, col);
        for(int i = 0; i < n->nb ; i++){
            printf("%f ", n->pBelief[i]);
        }
        printf("\n");

    }

    void setParentBeliefDamping(float gamma){
        if(gamma < 0 || gamma > 1.0){
            throw std::domain_error("setParentBeliefDamping: gamma must be between 0 and 1");
        }
        for(int n = 0; n < destin->nNodes ; n++){
            destin->nodes[n].gamma = gamma;
        }
    }

    void setPreviousBeliefDamping(float lambda){
        if(gamma < 0 || gamma > 1.0){
            throw std::domain_error("setParentBeliefDamping: lambda must be between 0 and 1");
        }
        for(int n = 0; n < destin->nNodes ; n++){
            destin->nodes[n].nLambda = lambda;
        }

    }

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
    void load(const char * fileName){
        if(centroidImages != NULL){
            Cig_DestroyCentroidImages(destin,  centroidImages);
            centroidImages = NULL;
        }

        destin = LoadDestin(destin, fileName);
        if(destin == NULL){
            throw std::runtime_error("load: could not open destin file.\n");
        }


        Node * n = GetNodeFromDestin(destin, 0, 0, 0);
        this->beta = n->beta;
        this->gamma = n->gamma;
        this->isUniform = destin->isUniform;
        this->lambda = n->nLambda;

        if(temperatures != NULL){
            delete [] temperatures;
        }
        temperatures = new float[getLayerCount()];
        memcpy(temperatures, destin->temp, sizeof(float) * getLayerCount());

    }

    /** Moves the given centroid to its node's input last observation.
      * @param layer
      * @param row
      * @param col
      * @param centroid - centroid to move
      */
    void moveCentroidToInput(int layer, int row, int col, int centroid){
        Node * n = getNode(layer, row, col);
        if(centroid >= destin->nb[layer]){
            throw std::domain_error("moveCentroidToInput: centroid out of bounds.");
        }
        float * cent = &n->mu[centroid * n->ns];
        memcpy(cent, n->observation, n->ns * sizeof(float));
    }

    /** Helps determine the centroid image contrast.
      * Values greater than 1.0 will provide more contrasts to
      * the centroid images generated via displayCentroidImage() method.
      */
    void setCentImgWeightExponent(float exp){
        centroidImageWeightParameter = exp;
    }


    void updateCentroidImages(){
        if(centroidImages==NULL){
            centroidImages = Cig_CreateCentroidImages(destin, centroidImageWeightParameter);
        }else{
            Cig_UpdateCentroidImages(destin, centroidImages, centroidImageWeightParameter);
        }
    }

    /** Returns the centroid image as a float array.
      * Then displays it.
      */
    float * getCentroidImage(int layer, int centroid){
        if(!destin->isUniform){
            printf("getCentroidImage: must be uniform");
            return NULL;
        }

        displayCentroidImage(layer, centroid);
        return centroidImages[layer][centroid];
    }

    /** Gets the generated centroid images as an opencv Mat image.
      */
    cv::Mat getCentroidImageM(int layer, int centroid, int disp_width = 256, bool enhanceContrast = false){
        if(!isUniform){
            throw std::logic_error("can't displayCentroidImage with non uniform DeSTIN.");
        }
        if(layer > destin->nLayers ||  centroid > destin->nb[layer]){
            throw std::domain_error("displayCentroidImage: layer, centroid out of bounds\n");
        }

        if(centroidImages == NULL){
            centroidImages = Cig_CreateCentroidImages(destin, centroidImageWeightParameter);
        }

        uint width = Cig_GetCentroidImageWidth(destin, layer);

        //initialize or create new grid if needed
        if(centroidImage.rows != width || centroidImage.cols != width){
            centroidImage = cv::Mat(width, width, CV_32FC1);
        }
        float * data = (float *)centroidImage.data;

        memcpy(data, centroidImages[layer][centroid], width*width*sizeof(float));

        cv::Mat toShow;
        centroidImage.convertTo(toShow, CV_8UC1, 255); // make it suitable for equalizeHist
        if(enhanceContrast){
            cv::equalizeHist(toShow, toShow);
        }

        cv::resize(toShow,centroidImageResized, cv::Size(disp_width, disp_width), 0, 0, cv::INTER_NEAREST);

        return centroidImageResized;
    }

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
    void displayCentroidImage(int layer, int centroid, int disp_width = 256, bool enhanceContrast = false, string window_name="Centroid Image" ){
        getCentroidImageM(layer, centroid, disp_width, enhanceContrast);
        cv::imshow(window_name.c_str(), centroidImageResized);
    }

    void saveCentroidImage(int layer, int centroid, string filename, int disp_width = 256, bool enhanceContrast = false){
        getCentroidImageM(layer, centroid, disp_width, enhanceContrast);
        cv::imwrite(filename, centroidImageResized);
    }

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
                                    ){
        if(!isUniform){
            throw std::logic_error("can't displayLayerCentroidImages with non uniform DeSTIN.");
        }

        if(centroidImages == NULL){
            centroidImages = Cig_CreateCentroidImages(destin, centroidImageWeightParameter);
        }

        int images = getBeliefsPerNode(layer);
        int images_wide = ceil(sqrt(images));
        int sub_img_width = (int)((double)scale_width / (double)images_wide - (double)border_width);

        int wpb = sub_img_width + border_width; // sub image width plus boarder. Each image gets a right and bottom boarder only.

        int images_high = ceil((float)images / (float)images_wide);

        cv::Mat big_img = cv::Mat::zeros(wpb*images_high, wpb*images_wide, CV_32FC1);

        int r, c, x, y;
        // copies the subimages into the correct place in the big image
        for(int i = 0 ; i < images ; i++){
                r = i  / images_wide;
                c = i - r * images_wide;
                x = c * wpb;
                y = r * wpb;
                int w = Cig_GetCentroidImageWidth(destin, layer);
                cv::Mat subimage(w, w, CV_32FC1, centroidImages[layer][i]);
                cv::Mat subimage_resized;
                cv::resize(subimage, subimage_resized, cv::Size(sub_img_width, sub_img_width), 0,0,cv::INTER_NEAREST);
                cv::Rect roi( cv::Point( x, y ), subimage_resized.size() );
                cv::Mat dest = big_img( roi );

                subimage_resized.copyTo( dest );
        }

        layerCentroidsImage = big_img;
        cv::imshow(window_title, layerCentroidsImage);
    }

};


#endif /* DESTINNETWORKALT_H_ */
