

#ifndef DESTINTREEMANAGER_H
#define DESTINTREEMANAGER_H


#include "DestinNetworkAlt.h"
class DestinTreeManager {

    DestinNetworkAlt & destin;
    short * winnerTree;
    int winningTreeSize;
    uint labelBucket;
    uint childNumBucket;
    int bottomLayer;
    const int nLayers;


    int buildTree(const Node * parent, int pos, const int child_num);

    void paintCentroidImage(int cent_layer, int centroid, int x, int y, cv::Mat & img);

    void calcChildCoords(int px, int py, int child_no, int child_layer, int & child_x_out, int & child_y_out);

public:

    DestinTreeManager(DestinNetworkAlt & destin, int bottom);

    ~DestinTreeManager();

    void decodeLabel(short label, int & cent_out, int & layer_out, int & child_num_out);

    short getTreeLabelForCentroid(const int centroid, const int layer, const int child_num);

    short * getWinningCentroidTree();

    int getWinningCentroidTreeSize(){
        return winningTreeSize;
    }

    void setBottomLayer(unsigned int bottom);

    void displayTree(std::vector<short> & tree);

    cv::Mat getTreeImg(std::vector<short> & tree);
};


#endif
