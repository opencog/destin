#ifndef CM_ORDERED_TREE_MINER_WRAPPER_H
#define CM_ORDERED_TREE_MINER_WRAPPER_H

#include <vector>
#include "PatternTree.h"
#include "OccLongList.h"
#include <string>

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
public:

    CMOrderedTreeMinerWrapper():current_tid(0), MIN_VERTEX((1 << 15) - 1), MAX_VERTEX(0)
    {

    }

    void addTree(short description[], int length){
        int total = length;
        short temp;
        stack<short> tempK;

        TextTree rhs; // have no clue why its called rhs

        rhs.tid = current_tid++;

        temp = description[0]; //read in the root label

        rhs.vLabel.push_back(temp);
        rhs.firstChild.push_back(-1); //temporarily, the root has no child
        rhs.nextSibling.push_back(-1); //the root has no sibling
        rhs.parent.push_back(-1); //the root has no parent
        rhs.vNumber = 1;

        tempK.push(0); //the index of the root

        for ( short i = 1; i < total; i++ ) {
            temp = description[i];
            if ( temp == -1 ) { //a backtrack
                tempK.pop();
                continue; //nothing to do with the TextTree
            }

            rhs.vLabel.push_back(temp); //add the new vertex label

            if (rhs.firstChild[tempK.top()] == -1) { //if the current node has no child yet
                rhs.firstChild[tempK.top()] = rhs.vNumber;
            }
            else { //if the current node has children already, find the rightmost child, its nextSibling is the new node
                short j = tempK.top();
                j = rhs.firstChild[j];
                while ( rhs.nextSibling[j] != -1 ) j = rhs.nextSibling[j];
                rhs.nextSibling[j] = rhs.vNumber;
            }

            rhs.firstChild.push_back(-1); //the new node has no child yet
            rhs.nextSibling.push_back(-1); //the new node has no right sibling yet
            rhs.parent.push_back(tempK.top()); //the parent of the new node is the current node
            tempK.push(rhs.vNumber);
            rhs.vNumber++;
        }

        database.push_back(rhs);

        for(int i = 0 ; i < rhs.vLabel.size() ; i++){
            short l = rhs.vLabel[i];
            if(l < MIN_VERTEX){
                MIN_VERTEX = l;
            }else if(l > MAX_VERTEX){
                MAX_VERTEX = l;
            }
        }
        return;
    }

    void mine(int support, vector<PatternTree> & maximal_out){

        currentPatternTree.initialSize();
        maximal_out.clear();
        vector<int> checked(1000,0); //TODO: what does 1000 do?
        vector<int> closed(1000,0);
        vector<int> maximal(1000,0);
        /******************************************************************
        step2.1: scan the database once, find frequent node labels
        ******************************************************************/
        vector<bool> isFrequent(MAX_VERTEX - MIN_VERTEX + 1, false);
        map<short,int> count;
        map<short,int>::iterator pos;


        for ( int i = 0; i < database.size(); i++ ) {
            vector<bool> isVisited(MAX_VERTEX - MIN_VERTEX + 1, false);
            for ( short j = 0; j < database[i].vNumber; j++ ) {
                short temp = database[i].vLabel[j] - MIN_VERTEX;
                if ( !isVisited[temp] ) {
                    isVisited[temp] = true;
                    pos = count.find(temp);
                    if ( pos != count.end() )
                        count[temp]++;
                    else
                        count.insert(make_pair(temp, 1));
                }
            }
        }
        for ( int i = 0; i < isFrequent.size(); i++ ) {
            if ( count[i] >= support ) isFrequent[i] = true;
        }

        /******************************************************************
        step2.2: scan the database another time, to get occurrenceList for all
        frequent nodes
        ******************************************************************/
        map<short,OccLongList> occLongList;
        map<short,OccLongList>::iterator pos2;
        vector<short> dummy;
        for ( int i = 0; i < database.size(); i++ ) {
            for ( short j = 0; j < database[i].vNumber; j++ ) {
                if ( isFrequent[database[i].vLabel[j] - MIN_VERTEX] == true ) {
                    occLongList[database[i].vLabel[j] - MIN_VERTEX].insert(i,dummy,j);
                }
            }
        }

        /******************************************************************
        step2.3: explore each frequent item
        ******************************************************************/
        for ( pos2 = occLongList.begin(); pos2 != occLongList.end(); ++pos2 ) {
            if ( pos2->second.mySupport >= support ) {
                currentPatternTree.addRightmost(pos2->first + MIN_VERTEX,0);
                pos2->second.explore(isFrequent,database,support,checked,closed,maximal, MIN_VERTEX, currentPatternTree, maximal_out);
                currentPatternTree.deleteRightmost();
            }
        }
        return;
    }

    void treeToVector(TextTree & tt, vector<short> & v_out){
         dfsVisit(0, tt, v_out);
         v_out.pop_back(); // remove the ending "-1"
         return;
    }

};

#endif
