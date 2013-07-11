
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
        winnerTree = new short[getWinningCentroidTreeSize()]; // deleted in deconstructor
    }

    buildTree(destin.getNode(nLayers - 1, 0, 0), 0, 0);
    return winnerTree;
}

vector<short> DestinTreeManager::getWinningCentroidTreeVector(){
    short * tree = getWinningCentroidTree();
    short * end = tree + getWinningCentroidTreeSize(); // pointer arrithmetic to get to end of the array
    vector<short> winTreeVect(tree, end); // copy array to the vector
    return winTreeVect;
}

int DestinTreeManager::buildTree(const Node * parent, int pos, const int child_num){
    int layer = parent->layer;
    winnerTree[pos] = getTreeLabelForCentroid(parent->winner, layer, child_num);
    if(layer > bottomLayer && parent->children != NULL){
        for(int i = 0 ; i < 4 ; i++){
            pos = buildTree(parent->children[i], ++pos, i);
            winnerTree[++pos] = -1;
        }
    }
    return pos;
}

void DestinTreeManager::createConvertNodeLocations(const Node * parent){
    int layer = parent->layer;
    NodeLocation nl;
    nl.col = parent->col;
    nl.row = parent->row;
    nl.layer = layer;
    convertNodeLocation.push_back(nl);
    if(layer > bottomLayer && parent->children != NULL){
        for(int i = 0 ; i < 4 ; i++){
            createConvertNodeLocations(parent->children[i]);
        }
    }
    return;
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

    int nodes_used = destin.getNetwork()->nNodes - nodes_to_subtract;
    winningTreeSize = (nodes_used - 1) * 2 + 1;

    if(winnerTree!=NULL){
        delete [] winnerTree;
        winnerTree = NULL;
    }

    convertNodeLocation.reserve(nodes_used);
    convertNodeLocation.clear();
    createConvertNodeLocations(destin.getNode(nLayers - 1, 0, 0));

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

void DestinTreeManager::displayFoundSubtree(const int treeIndex){
    vector<short> tree;
    if(treeIndex >= foundSubtrees.size()){
        std::cerr << "displayMinedTree: index out of bounds." << std::endl;
    }
    tmw.treeToVector(foundSubtrees.at(treeIndex), tree);
    displayTree(tree);
}

void DestinTreeManager::saveFoundSubtreeImg(const int treeIndex, const string & filename){
    vector<short> tree;
    tmw.treeToVector(foundSubtrees.at(treeIndex), tree);
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
    foundSubtrees.clear();
    tmw.mine(support, foundSubtrees);
    return foundSubtrees.size();
}

std::vector<short> DestinTreeManager::getFoundSubtree(const int treeIndex){
    vector<short> out;
    tmw.treeToVector(foundSubtrees.at(treeIndex), out);
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

string DestinTreeManager::getFoundSubtreeAsString(const int treeIndex){
    vector<short> t;
    stringstream ss;
    tmw.treeToVector(foundSubtrees.at(treeIndex), t);
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

void DestinTreeManager::printFoundSubtree(const int treeIndex){
    cout << getFoundSubtreeAsString(treeIndex);
    return;
}

void DestinTreeManager::timeShiftTrees(){
    tmw.timeShiftDatabase(destin.getLayerCount());
}

vector<int> DestinTreeManager::matchSubtree(int foundSubtreeIndex){
    vector<int> matches;
    for(int i = 0 ; i < getAddedTreeCount() ; i++){
        if(tmw.isSubTreeOf(tmw.getAddedTree(i), foundSubtrees.at(foundSubtreeIndex))){
            matches.push_back(i);
        }
    }
    return matches;
}

void DestinTreeManager::drawSubtreeBordersOntoImage(int foundSubtree, cv::Mat & canvas, bool justOne, int thickness){
    TextTree & needle = foundSubtrees.at(foundSubtree);
    TextTree haystack;
    tmw.arrayToTextTree(getWinningCentroidTree(), getWinningCentroidTreeSize(), 0, haystack);
    vector<int> locations;
    if(justOne){
        int location = tmw.findSubtreeLocation(haystack, needle);
        if(location != -1){
            locations.push_back(location);
        }
    }else{
        locations = tmw.findSubtreeLocations(haystack, needle);
    }

    /// this block draws the border of the found subtree
    cv::Scalar blue(255,0,0); //BGR color
    uint * layer_widths =  destin.getNetwork()->layerWidth;
    int imageWidth = destin.getInputImageWidth();
    for(int i = 0 ; i < locations.size() ; i++){
        // find the node location of the subtree
        NodeLocation & nl = convertNodeLocation.at(locations[i]);

        // create the box around the subtree
        int box_width = imageWidth / layer_widths[nl.layer]; // how wide the box is
        cv::Rect box(nl.col * box_width, nl.row * box_width, box_width, box_width);

        // draw the box on the canvas image
        cv::rectangle(canvas, box, blue, thickness);
    }

    return;
}

void DestinTreeManager::displayFoundSubtreeBorders(int foundSubtree, cv::Mat & canvas, bool justOne, int thickness, int waitkey_delay, const string & winname){
    drawSubtreeBordersOntoImage(foundSubtree, canvas, justOne, thickness);
    cv::imshow(winname, canvas);
    if(waitkey_delay != 0){
        cv::waitKey(waitkey_delay);
    }
}
