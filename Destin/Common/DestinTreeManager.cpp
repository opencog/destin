
#include "DestinTreeManager.h"

DestinTreeManager::DestinTreeManager(DestinNetworkAlt & destin, int bottom)
:destin(destin), nLayers(destin.getLayerCount()), winnerTree(NULL)
{
    labelBucket = ( 1 << ( sizeof(short) * 8 - 1))/nLayers;
    childNumBucket = labelBucket / 4;
    setBottomLayer(bottom);
    for(int i = 0 ; i <  nLayers; i++){
        if( destin.getBeliefsPerNode(i) >= childNumBucket){
            throw std::domain_error("DestinTreeManager: too many centroids\n.");
        }
    }
}

DestinTreeManager::~DestinTreeManager(){
    if(winnerTree!=NULL){
        delete [] winnerTree;
        winnerTree = NULL;
    }
}

void DestinTreeManager::decodeLabel(short label, int & cent_out, int & layer_out, int & child_num_out){
       layer_out = label / labelBucket;
       int temp = label -  labelBucket * layer_out;
       child_num_out = temp / childNumBucket;
       cent_out = temp - child_num_out * childNumBucket;
}

short DestinTreeManager::getTreeLabelForCentroid(const int centroid, const int layer, const int child_num){
    return layer * labelBucket + childNumBucket * child_num + centroid;
}


short * DestinTreeManager::getWinningCentroidTree(){
    if(!destin.getNetwork()->isUniform){
        throw std::logic_error("DestinTreeManager::getWinningCentroidTree only uniform destin is supported.\n");
    }
    if(winnerTree==NULL){
        winnerTree = new short[getWinningCentroidTreeSize()];
    }

    buildTree(destin.getNode(nLayers - 1, 0, 0), 0, 0);
    return winnerTree;
}

int DestinTreeManager::buildTree(const Node * parent, int pos, const int child_num){
    winnerTree[pos] = getTreeLabelForCentroid(parent->winner, parent->layer, child_num);
    if(parent->layer > bottomLayer && parent->children != NULL){
        for(int i = 0 ; i < 4 ; i++){
            pos = buildTree(parent->children[i], ++pos, i);
            winnerTree[++pos] = -1;
        }
    }
    return pos;
}

void DestinTreeManager::setBottomLayer(unsigned int bottom){
    if(bottom >= destin.getNetwork()->nLayers){
        throw std::domain_error("setBottomLayer: cannot be set to above the top layer\n");
    }

    bottomLayer = bottom;
    Destin * d = destin.getNetwork();

    uint nodes_to_subtract = 0;
    for(int i = 0 ; i < bottom ; i++){
        nodes_to_subtract += d->layerSize[i];
    }

    winningTreeSize = (destin.getNetwork()->nNodes - nodes_to_subtract - 1) * 2 + 1;

    if(winnerTree!=NULL){
        delete [] winnerTree;
        winnerTree = NULL;
    }
    return;
}

cv::Mat DestinTreeManager::getTreeImg(const std::vector<short> & tree){
    if(tree.size()==0){
        std::cerr << " DestinTreeManager::getTreeImg: tree was empty.\n";
        return cv::Mat();
    }

    int layer, cent, child_num;
    decodeLabel(tree[0], cent, layer,  child_num);
    int w = Cig_GetCentroidImageWidth(destin.getNetwork(), layer);
    cv::Mat img = cv::Mat::zeros(w, w, CV_32FC1); //initialize the image to black

    std::vector<int> xs; // x stack
    std::vector<int> ys; // y stack

    int x=0, y=0;
    for(int i = 0 ;  i < tree.size() ; i++){
        int label = tree[i];

        if(label != -1){
            decodeLabel(label, cent, layer, child_num); //todo: what if its the root of subtree but not of top node of destin
            if(i > 0){
                calcChildCoords(xs.back(), ys.back(), child_num, layer, x, y);
            }
            paintCentroidImage(layer, cent, x, y, img );
            xs.push_back(x);
            ys.push_back(y);
        }else{
#ifdef UNIT_TEST
            if(x_stack.empty()){
                throw std::runtime_error("DestinTreeManager::displayTree: trying to pop empty vector\n");
            }
#endif
            xs.pop_back();
            ys.pop_back();
        }
    }


    return img;
}

void DestinTreeManager::displayTree(const std::vector<short> & tree){
    if(tree.size()==0){
        std::cerr << "displayTree: tree was empty.\n";
        return;
    }
    cv::Mat img = getTreeImg(tree);
    cv::imshow("display tree", img);
    return;
}

void DestinTreeManager::displayMinedTree(const int treeIndex){
    vector<short> tree;
    tmw.treeToVector(minedTrees.at(treeIndex), tree);
    displayTree(tree);
}

void DestinTreeManager::saveMinedTreeImg(const int treeIndex, const string & filename){
    vector<short> tree;
    tmw.treeToVector(minedTrees.at(treeIndex), tree);
    cv::Mat img = getTreeImg(tree);
    cv::Mat towrite;
    img.convertTo(towrite, CV_8UC1, 255);
    cv::imwrite(filename, towrite);
    return;
}

void DestinTreeManager::paintCentroidImage(int cent_layer, int centroid, int x, int y, cv::Mat & img){
    int w = Cig_GetCentroidImageWidth(destin.getNetwork(), cent_layer); //TODO: what is smallest depth
    cv::Mat subimage(w, w, CV_32FC1,destin.getCentroidImages()[cent_layer][centroid]); // wrap centroid image with cv::Mat
    cv::Rect roi(cv::Point( x, y ), subimage.size()); // make roi = region of interest
    cv::Mat dest = img( roi );     // get the subsection of img where the centroid image will go
    subimage.copyTo(dest);                                                  // copy the subimage to the subsection of img
    return;
}

void DestinTreeManager::calcChildCoords(int px, int py, int child_no, int child_layer, int & child_x_out, int & child_y_out){
    int w;
    switch (child_no) {
        case 0:
            child_x_out = px;
            child_y_out = py;
            break;
        case 1:
            w = Cig_GetCentroidImageWidth(destin.getNetwork(), child_layer);
            child_x_out = px + w;
            child_y_out = py;
            break;
        case 2:
            w = Cig_GetCentroidImageWidth(destin.getNetwork(), child_layer);
            child_x_out = px;
            child_y_out = py + w;
            break;
        case 3:
            w = Cig_GetCentroidImageWidth(destin.getNetwork(), child_layer);
            child_x_out = px + w;
            child_y_out = py + w;
            break;
        default :
            throw std::domain_error("DestinNetworlAlt::calcNewCoords: invalid child number.");
    };
    return;
}

void DestinTreeManager::addTree(){
    tmw.addTree(getWinningCentroidTree(), getWinningCentroidTreeSize());
}

int DestinTreeManager::mine(const int support){
    minedTrees.clear();
    tmw.mine(support, minedTrees);
    return minedTrees.size();
}

std::vector<short> DestinTreeManager::getMinedTree(const int treeIndex){
    vector<short> out;
    tmw.treeToVector(minedTrees.at(treeIndex), out);
    return out;
}

void DestinTreeManager::printHelper(TextTree & pt, short vertex, int level, stringstream & ss){
    for (int i = 0; i < level; i++ ){
        ss << '\t';
    }
    int cent, layer, pos;
    this->decodeLabel(pt.vLabel[vertex], cent, layer, pos);
    ss << "(L"  << layer << ",C" << cent << ",P" << pos << ")" << endl;
    int child = pt.firstChild.at(vertex);
    if(child !=-1){
        printHelper(pt, child, level + 1, ss);
        int sib = pt.nextSibling.at(child);
        while(sib != -1){
            printHelper(pt, sib, level + 1, ss);
            sib = pt.nextSibling.at(sib);
        }
    }

    return ;
}

string DestinTreeManager::getMinedTreeAsString(const int treeIndex){
    vector<short> t;
    stringstream ss;
    tmw.treeToVector(minedTrees.at(treeIndex), t);
    ss << "size: " << t.size() << " : ";
    for(int i = 0 ; i < t.size() ; i ++){
        int cent, layer, pos;
        if(t[i] != -1){
            this->decodeLabel(t[i], cent, layer, pos);
            ss << "(L" << layer << ",C" << cent << ",P" << pos <<")";
        }else{
            ss << " (GoUp) ";
        }
    }
    ss << endl;
    return ss.str();
}

void DestinTreeManager::printMinedTree(const int treeIndex){
    cout << getMinedTreeAsString(treeIndex);
    return;
}

void DestinTreeManager::timeShiftTrees(){
    tmw.timeShiftDatabase(destin.getLayerCount());
}
