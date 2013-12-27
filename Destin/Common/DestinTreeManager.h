#ifndef DESTINTREEMANAGER_H
#define DESTINTREEMANAGER_H


#include "DestinNetworkAlt.h"
#include "CMOrderedTreeMinerWrapper.h"

/**
 * Performs subtree mining using CMOrderedTreeMinerWrapper.
 * Is able to display the found mined trees.
 * Currently only supports a mix of 4 children to 1 parent node destin heirarchies,
 * or parent_layer_width = child_layer_width + 1 heirarchy.
 */
class DestinTreeManager {

    DestinNetworkAlt & destin;
    short * winnerTree;
    int winningTreePathSize; // length of winning tree depth first search path. Is # nodes plus "-1" tracebacks.
    uint labelBucket; // divide the range of short into nLayer bucket ranges
    uint childNumBucket;
    uint maxChildrenPerParent;
    int bottomLayer;
    const int nLayers;

    CMOrderedTreeMinerWrapper tmw; //tree miner wrapper
    vector<PatternTree> foundSubtrees; //list of trees that were found during subtree mining.

    struct NodeLocation {
        int layer;
        int row;
        int col;
    };

    // converts a depth first path (without -1 backtraces) index into a particular node location.
    // Used while drawing borders around subtree locations.
    vector<NodeLocation> convertNodeLocation;

    // contructs convertNodeLocation vector
    void createConvertNodeLocationsHelper(const Node * parent);

    // helper method for getWinningCentroidTree()
    int buildTree(const Node * parent, int pos, const int child_num);

    // helper method for getTreeImg()
    void paintCentroidImage(int cent_layer, int centroid, int x, int y, cv::Mat & img);

    // helper method for getTreeImg()
    void calcChildCoords(uint childrenWidth, int px, int py, int child_no, int child_layer, int & child_x_out, int & child_y_out);

    // helper for getMinedTreeStructureAsString
    void printHelper(TextTree & pt, short vertex, int level, stringstream & ss);

    // pre calculates the length of the array returned from getWinningCentroidTree()
    void calcWinningCentroidTreeSize(const Node * parent, int & count_out);

    void updateTreeParameters(int);

    void createConvertNodeLocations(int bottom_layer);

    int winningTreeNodeSize(int bottom_layer);

public:

    /** Constructor
      * @param destin - which destin network to wrap
      * @param bottom - how deep the extracted destin trees are.
      * Integer from 0 to number of layers - 1.
      * If 0 then the whole destin heirachy is used. If bottom = # of layers - 1, then
      * only a tree with depth 1, i.e. just the root node is used.
      * Smaller trees are mined faster.
      */
    DestinTreeManager(DestinNetworkAlt & destin, int bottom);

    ~DestinTreeManager();

    /** Takes the given label and decodes it into centroid, layer, and child position ( 0 to 3 )
      */
    void decodeLabel(short label, int & cent_out, int & layer_out, int & child_pos_out);

    /** Takes a layer, centroid number and child position and encodes it into one tree label.
      * @return - the encoded tree label.
      */
    short getTreeLabelForCentroid(const int centroid, const int layer, const int child_position);

    /** Gets a tree of the winning centroid indexes of the destin network, represented a list.
      * Encodes the tree by a depth first search path, using the getTreeLabelForCentroid()
      * method to get the label for each node, and using a -1 to represent a traceback.
      *
      * The tree will only go down to and including the bottom layer as set by setBotomLayer(int)
      */
    short * getWinningCentroidTree();

    /** Wraps array from getWinningCentroidTree() into a vector
     */
    vector<short> getWinningCentroidTreeVector();

    /** Returns the length of the tree.
      * Number of destin nodes plus the number of back traces.
      * An overlapping nodes destin heirarchy will be larger
      * than expected because each node can be visted serveral times.
      * Length of the array returned from getWinningCentroidTree().
      */
    int getWinningCentroidTreeSize(){
        return winningTreePathSize;
    }

    /** Sets how deep the mined trees are.
      * See bottom param of the constructor.
      */
    void setBottomLayer(unsigned int bottom);

    /** Creates a representative image of the given tree.
      * It uses DestinNetowrkAlt::getCentroidImage
      * to get the image for the root, which is the entire size of
      * the of the output image. In a depth first recursive fashion, then children of
      * the root are drawn next, but the children images only cover up a fraction of their parent's
      * image. The deeper the child image is in the tree, the smaller its overwriting image is.
      * The location of child images are determined by their position in the tree,
      * and relative to their parent images.
      * @param tree - the tree to get the image for. See CMOrderedTreeMinerWrapper::addTree method for the format.
      * @return the image as an opencv Mat
      *
      */
    cv::Mat getTreeImg(const std::vector<short> & tree);

    /** Displays the image given tree generated with getTreeImg()
      * cv::waitKey must be called after for it to show.
      * @param tree - the tree to get the image for. See CMOrderedTreeMinerWrapper::addTree method for the format.
      */
    void displayTree(const std::vector<short> & tree);

    /** Performs subtree mining.
      * Mines on the trees that were added with the addTree() method.
      * Previously found trees are cleared from the internal list before hand.
      * @param support - support used in mining. Only subtrees that occur
      * at least this amount of times are found.
      * @return - the number of minded trees found
      */
    int mine(const int support);

    /** Adds a tree to be mined with mine() method.
      * The added tree is taken from a call to getWinningCentroidTree()
      *
      */
    void addTree();

    /** Returs how many trees have been added by addTree()
      */
    int getAddedTreeCount(){
        return tmw.getAddedTreeCount();
    }

    /** Uses displayTree() method to show the given found mined tree.
      * cv::waitKey must be called after for it to show.
      * @param treeIndex - which tree to display
      */
    void displayFoundSubtree(const int treeIndex);


    /** Saves an image of the given found mined subtree to a file.
      */
    void saveFoundSubtreeImg(const int treeIndex, const string & filename);

    /** Returns a copy of the given mined tree found from mine() method.
      */
    std::vector<short> getFoundSubtree(const int treeIndex);

    /** Returns how many frequent subtrees have been found from mine() method.
      */
    int getFoundSubtreeCount(){
        return foundSubtrees.size();
    }

    /** Prints the given mined tree as a depth first search path list
      *  For example:
      *  size: 7 : (L6,C2,P2) (L5,C4,P0)  (GoUp) (L5,C9,P2)  (GoUp) (L5,C9,P3)  (GoUp)
      */
    void printFoundSubtree(const int treeIndex);

    string getFoundSubtreeAsString(const int treeIndex);

    /** Prints the given mined tree in a treelike fashion
      * For example:
      *        (L6,C2,P2)
      *            (L5,C4,P0)
      *            (L5,C9,P2)
      *            (L5,C9,P3)
      *  L = Level, C = centroid, P = child position ( 0 to 3 )
      */
    void printFoundSubtreeStructure(const int treeIndex){
        cout << getFoundSubtreeStructureAsString(treeIndex);
    }

    /** Same as printFoundSubtreeStructure but returns the string
     * instead of printing to stdout.
     */
    string getFoundSubtreeStructureAsString(const int treeIndex){
        stringstream ss;
        printHelper(foundSubtrees.at(treeIndex), 0, 0, ss);
        return ss.str();
    }

    /** Same as printMinedTreeStructure but shows trees
      * in the original tree database instead of mined subtrees.
      */
    void printAddedTreeStructure(const int treeIndex){
        stringstream ss;
        printHelper(tmw.getAddedTree(treeIndex), 0, 0, ss);
        cout << ss.str();
    }


    /** Returns the underlying tree mining object
      */
    CMOrderedTreeMinerWrapper & getTreeMiner(){
        return tmw;
    }

    /** Constructs new trees that each represent one time slice.
      *
      * This adjusts trees that have been added with addTree() so that
      * each tree will represent one instance in time. It does
      * this by looking ahead to future trees to construct the "present" tree.
      *
      * Normaly it takes N destin cycles, where N = number of layers
      * in the destin heirarcy, before an input image makes it way from the
      * bottom layer to the top layer and so all winning centroid trees
      * will represent different instances of time. Calling this method
      * will make it look like the images fed through destin move through
      * it instanly without the time delays.
      *
      * After this call, the last N - 1 trees will be thrown out because of
      * lack of enough trees to look ahead with
      *
      * @throws std::runtime_error if there are not enough trees to make at least
      * one correct tree.
      */
    void timeShiftTrees();

    /** Determines what database trees the given frequent subtree is found in.
      * @param foundSubtreeIndex - searches for this found subtree in the tree database.
      * It must be in range of the count given by getMinedTreeCount().
      * @return list of indicies of trees, in the tree database, that the
      *         given found frequent subtree is a subtree of.
      */
    vector<int> matchSubtree(int foundSubtreeIndex);

    /** Draws the border box ( in blue ) of the given subtree onto the canvas image.
     * The location and size of the box is determined by where the selected found subtree
     * is currently located in the current winning centroid tree ( given by a call
     * to the getWinningCentroidTree method ).
     *
     * @param foundMinedSubtree - index of one of the subtrees found from calling the mine() method which will be
     * searched for in the current winning centroid tree.
     * @param canvas - the image to draw the border box on. Should be a color matrix.
     * @param justOne - if true, then only the first location the subtree is found is drawn with a box, otherwise
     * multiple border boxes will be drawn for each location the subtree is found.
     * @param thickness - the thickness of the box border. Default is 2.
     */
    void drawSubtreeBordersOntoImage(int foundMinedSubtree, cv::Mat & canvas, bool justOne = true, int thickness = 2);

    /** Displays the canvas image after it has been drawn on by the drawSubtreeBordersOntoImage method.
     * @see DestinTreeManager::drawSubtreeBordersOntoImage
     *
     * @param foundMinedSubtree - index of one of the subtrees found from calling the mine() method which will be
     * searched for in the current winning centroid tree.
     * @param canvas - the image to draw the border box on. Should be a color matrix.
     * @param justOne - if true, then only the first location the subtree is found is drawn with a box, otherwise
     * multiple border boxes will be drawn for each location the subtree is found.
     * @param thickness - the thickness of the box border. Default is 2.
     * @param waitkey_delay - the delay in milliseconds passed to the cv::waitKey() function so that the
     *  image will be refreshed. If 0 is passed, then cv::waitKey() will not be called. Default is 300 milliseconds.
     * @param winname - The name of the window that displays the image.
     */
    void displayFoundSubtreeBorders(int foundMinedSubtree, cv::Mat & canvas, bool justOne = true, int thickness = 2, int waitkey_delay = 300, const string & winname = "Tree borders");

    /** Clears the list of added trees and list of found subtrees.
      */
    void reset(){
        foundSubtrees.clear();
        tmw.reset();
    }

};


#endif
