#ifndef CM_ORDERED_TREE_MINER_WRAPPER_H
#define CM_ORDERED_TREE_MINER_WRAPPER_H

#include <vector>
#include <string>
#include <stdexcept>

#include "PatternTree.h"
#include "OccLongList.h"

using std::vector;



/** Defined in TextTree.cpp but isn't in a header
  */
void dfsVisit(const short current, const TextTree& rhs, vector<short>& zakiCode);


/**
 * @brief The CMOrderedTreeMinerWrapper class.
 * Wrapper for CMOrderedTreeMiner which we try to leave
 * as unmodified as possible.
 */
class CMOrderedTreeMinerWrapper {

    int current_tid;
    PatternTree currentPatternTree;
    vector<TextTree> database;
    short MAX_VERTEX, MIN_VERTEX;

    int isSubTreeOfHelper(const TextTree & parent_tree, const TextTree & child_tree, const short pt_vertex);

    /**
      * Assumes that all the trees in the database have all the same size and structure
      */
    void timeShiftDatabaseHelper(const short vertex, const int level, const int tree_index);

    void resetStats(){
        MIN_VERTEX = (1 << 15) - 1;
        MAX_VERTEX = 0;
        current_tid = 0;
    }

public:

    CMOrderedTreeMinerWrapper()
    {
        resetStats();
    }

    /** Returns how many trees have been added with addTree()
      */
    int getAddedTreeCount(){
        return database.size();
    }

    /** Adds a tree to the database to be mined.
      *
      * The tree is input as a depth first path array.
      *
      * For example, this 3 node tree
      *
      *      5
      *     / \
      *    3   2
      *
      * Would be input as an array [5, 3, -1, 2, -1].
      * The -1 entries represent a back track in the search path.
      * The search path needs to end at root, so it should end in one or more -1s
      * unless it's a one node tree, which in that case the search path will
      * be of length 1 with just the node.
      *
      * @param treeDescription - depth first path array of tree to add.
      * @param length - length of the treeDescription array.
      */
    void addTree(short treeDescription[], int length);

    TextTree & getAddedTree(int index){
        return database.at(index);
    }

    /** Mines the trees database added with addTree() to find
      * maximal, frequent, subtrees.
      *
      * @param support - subtrees must appear this many times to be considerd frequent
      * @param maximaml_out - this output list gets populated with the minded subtrees
      */
    void mine(int support, vector<PatternTree> & maximal_out);

    /** Converts a TextTree or PatternTree to a short vector.
      * @param tt - the TextTree or PatternTree to convert.
      * @param v_out - the output vector. See addTree() for the format of this vector
      */
    void treeToVector(TextTree & tt, vector<short> & v_out);


    void vectorToTextTree(vector<short> treeDesciption, int tid, TextTree & out);
    void arrayToTextTree(short description[], int length, int tid, TextTree & out);

    /** Adjusts the database so that each tree represents one time slice.
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
      * lack of enough trees to look ahead with.
      *
      * Assumes all trees in the database have the same depth and structure.
      *
      * @throws std::runtime_error if there are not enough trees to make at least
      * one correct tree.
      */
    void timeShiftDatabase(int treeDepth);


    /** Determines if the child tree is a copy of the parent tree and same structure
      * or only has some nodes missing, but not if the child tree nodes
      * are in the wrong order or has more nodes in certain places, or the
      * nodes have different labels.
      *
      * For example, if these two trees are given:
      * Parent tree:
      *   1
      *  /|\
      * 2 3 4
      *
      * Child tree:
      *   1
      *  / \
      * 2   4
      *
      * then it returns true.
      *
      * @param parent_tree
      * @param child_tree
      * @param pt_start_node - consider this node as the new root of the parent tree
      * @return position in the parent tree where the root node of the
      * child tree matches or -1 if it does not match. The position is the index
      * of the depth first path array of the parent tree with the back traces ( -1s ) removed.
      */
    int treeMatchesHelper(const TextTree & parent_tree,
                           const TextTree & child_tree,
                           const short pt_start_node);


    /** Determines if child tree (needle) is a subtree of parent tree (haystack)
      * @param haystack - searches in this containing tree
      * @param needle - searches for this subtree in the containing tree
     */
    bool isSubTreeOf(TextTree & haystack, TextTree & needle){
        return isSubTreeOfHelper(haystack, needle, 0) != -1;
    }

    /**
     * Same as findSubtreeLocations method but only returns the first found location.
     * @param haystack - Parent tree. Searches in this tree for the child subtree
     * @param needle - Child subtree. Search for this child subtree in the parent subtree.
     * @return Location in the parent tree of the root of the first match of the needle subtree.
     */
    int findSubtreeLocation(TextTree & haystack, TextTree & needle){
        return isSubTreeOfHelper(haystack, needle, 0);
    }

    /**
     * Finds the parent tree vertex locations where the child subtree is found.
     * Each of the parent vertex locations is at the root of a found child subtree. The parent
     * vertex locations are indicies of the parent tree's depth first array description with
     * the -1 backtraces removed.
     *
     * For example, if the child subtree is found in 4 different places of the parent subtree
     * then the returned list will be of size 4.
     *
     * @param haystack - Parent tree. Searches in this tree for the child subtree
     * @param needle - Child subtree. Search for this child subtree in the parent subtree.
     * @return list of parent tree verticies where the root of the child subtree is found.
     */
    vector<int> findSubtreeLocations(TextTree & haystack, TextTree & needle);

    /** Clears added and found trees.
      */
    void reset();
};

#endif
