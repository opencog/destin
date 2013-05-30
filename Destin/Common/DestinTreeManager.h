#ifndef DESTINTREEMANAGER_H
#define DESTINTREEMANAGER_H


#include "DestinNetworkAlt.h"
#include "CMOrderedTreeMinerWrapper.h"

/**
 * Performs subtree mining using CMOrderedTreeMinerWrapper.
 * Is able to display the found mined trees.
 */
class DestinTreeManager {

    DestinNetworkAlt & destin;
    short * winnerTree;
    int winningTreeSize;
    uint labelBucket;
    uint childNumBucket;
    int bottomLayer;
    const int nLayers;

    CMOrderedTreeMinerWrapper tmw; //tree miner wrapper
    vector<PatternTree> minedTrees;


    // helper method for getWinningCentroidTree()
    int buildTree(const Node * parent, int pos, const int child_num);

    // helper method for getTreeImg()
    void paintCentroidImage(int cent_layer, int centroid, int x, int y, cv::Mat & img);

    // helper method for getTreeImg()
    void calcChildCoords(int px, int py, int child_no, int child_layer, int & child_x_out, int & child_y_out);


    void printHelper(TextTree & pt, short vertex, int level, stringstream & ss);

public:

    /** Constructor
      * @param destin - which destin network to wrap
      * @param bottom - how deep the extracted destin trees are.
      * If 0 then the whole destin heirachy is used. If bottom = # of layers - 1, then
      * only a tree with depth 1, i.e. just the root node is used.
      * Smaller trees are mined faster.
      */
    DestinTreeManager(DestinNetworkAlt & destin, int bottom);

    ~DestinTreeManager();

    /** Takes the given label and decodes it into centroid, layer, and child position ( 0 to 3 )
      */
    void decodeLabel(short label, int & cent_out, int & layer_out, int & child_num_out);

    /** Takes a layer, centroid number and child position and encodes it into one tree label.
      * @return - the encoded tree label.
      */
    short getTreeLabelForCentroid(const int centroid, const int layer, const int child_num);

    /** Gets a tree of the winning centroid indexes of the destin network, represented a list.
      * Encodes the tree by a depth first search path, using the getTreeLabelForCentroid()
      * method to get the label for each node, and using a -1 to represent a traceback.
      */
    short * getWinningCentroidTree();

    /** Returns the length of the tree.
      * Length of the array returned from getWinningCentroidTree().
      */
    int getWinningCentroidTreeSize(){
        return winningTreeSize;
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
      * @return the image as an opencv Mat
      */
    cv::Mat getTreeImg(const std::vector<short> & tree);

    /** Displays the image given tree generated with getTreeImg()
      * cv::waitKey must be called after for it to show.
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
    void displayMinedTree(const int treeIndex);


    /** Saves an image of the given found mined subtree to a file.
      */
    void saveMinedTreeImg(const int treeIndex, const string & filename);

    /** Returns a copy of the given mined tree found from mine() method.
      */
    std::vector<short> getMinedTree(const int treeIndex);

    /** Returns how many frequent subtrees have been found from mine() method.
      */
    int getMinedTreeCount(){
        return minedTrees.size();
    }

    /** Prints the given mined tree as a depth first search path list
      *  For example:
      *  size: 7 : (L6,C2,P2) (L5,C4,P0)  (GoUp) (L5,C9,P2)  (GoUp) (L5,C9,P3)  (GoUp)
      */
    void printMinedTree(const int treeIndex);

    string getMinedTreeAsString(const int treeIndex);

    /** Prints the given mined tree in a treelike fashion
      * For example:
      *        (L6,C2,P2)
      *            (L5,C4,P0)
      *            (L5,C9,P2)
      *            (L5,C9,P3)
      *  L = Level, C = centroid, P = child position ( 0 to 3 )
      */
    void printMinedTreeStructure(const int treeIndex){
        cout << getMinedTreeStructureAsString(treeIndex);
    }

    string getMinedTreeStructureAsString(const int treeIndex){
        stringstream ss;
        printHelper(minedTrees.at(treeIndex), 0, 0, ss);
        return ss.str();
    }

    /** Same as printMinedTreeStructure but shows trees
      * in the database.
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
    vector<int> matchSubtree(int foundSubtreeIndex){
        vector<int> matches;
        for(int i = 0 ; i < getAddedTreeCount() ; i++){
            if(tmw.isSubTreeOf(tmw.getAddedTree(i), minedTrees.at(foundSubtreeIndex))){
                matches.push_back(i);
            }
        }
        return matches;
    }
};


#endif
