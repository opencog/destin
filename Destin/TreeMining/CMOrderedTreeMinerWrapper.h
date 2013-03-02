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


public:

    CMOrderedTreeMinerWrapper():current_tid(0), MIN_VERTEX((1 << 15) - 1), MAX_VERTEX(0)
    {

    }

    /** Returns how many trees have been added with addTree()
      */
    int getTreeCount(){
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
      * The -1 entries represent a trackback in the array.
      */
    void addTree(short description[], int length);

    TextTree & getAddedTree(int index){
        return database.at(index);
    }

    void mine(int support, vector<PatternTree> & maximal_out);

    void treeToVector(TextTree & tt, vector<short> & v_out);


    /**
      * Assumes that all the trees in the database have all the same size and structure
      */
    void timeShiftDatabaseHelper(const short vertex, const int level, const int tree_index);


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
