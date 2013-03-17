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


class CMOrderedTreeMinerWrapper {

    int support;
    int current_tid;
    PatternTree currentPatternTree;
    vector<TextTree> database;
    short MAX_VERTEX, MIN_VERTEX;

    bool isSubTreeOfHelper(const TextTree & parent_tree, const TextTree & child_tree, const short pt_vertex);

    /**
      * Assumes that all the trees in the database have all the same size and structure
      */
    void timeShiftDatabaseHelper(const short vertex, const int level, const int tree_index);

public:

    CMOrderedTreeMinerWrapper():current_tid(0), MIN_VERTEX((1 << 15) - 1), MAX_VERTEX(0)
    {
    }

    /** Returns how many trees have been added with addTree()
      */
    int getAddedTreeCount(){
        return database.size();
    }

    /** Adds a tree to the database to be mined.
      *
      * The tree is given a depth first search path.
      *
      * For example, this 3 node tree
      *
      *      5
      *     / \
      *    3   2
      *
      * Would be input as an array [5, 3, -1, 2, -1]
      * The -1 entries represent a back track in the search path.
      */
    void addTree(short description[], int length);

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


    /** @return true if the child tree is a copy of the parent tree and same structure
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
      * @param ct_start_node - consider this node as the new root of the child tree
      */
    bool treeMatchesHelper(const TextTree & parent_tree,
                           const TextTree & child_tree,
                           const short pt_start_node,
                           const short ct_start_node);


    /** Determines if child tree (needle) is a subtree of parent tree (haystack)
      * @param haystack - searches in this containing tree
      * @param needle - searches for this subtree in the containing tree
     */
    bool isSubTreeOf(TextTree & haystack, TextTree & needle){
        return isSubTreeOfHelper(haystack, needle, 0);
    }


};

#endif
