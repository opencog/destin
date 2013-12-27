#include "CMOrderedTreeMinerWrapper.h"



void CMOrderedTreeMinerWrapper::vectorToTextTree(vector<short> treeDescription, int tid, TextTree & out){
    arrayToTextTree(treeDescription.data(), treeDescription.size(), tid, out);
}

void CMOrderedTreeMinerWrapper::arrayToTextTree(short treeDescription[], int length, int tid, TextTree & out){
    stack<short> tempK;

    out.tid = tid;

    short temp = treeDescription[0]; //read in the root label

    out.vLabel.push_back(temp);
    out.firstChild.push_back(-1); //temporarily, the root has no child
    out.nextSibling.push_back(-1); //the root has no sibling
    out.parent.push_back(-1); //the root has no parent
    out.vNumber = 1;

    tempK.push(0); //the index of the root

    for ( short i = 1; i < length; i++ ) {
        temp = treeDescription[i];
        if ( temp == -1 ) { //a backtrack
            tempK.pop();
            continue; //nothing to do with the TextTree
        }

        out.vLabel.push_back(temp); //add the new vertex label

        if (out.firstChild[tempK.top()] == -1) { //if the current node has no child yet
            out.firstChild[tempK.top()] = out.vNumber;
        }
        else { //if the current node has children already, find the rightmost child, its nextSibling is the new node
            short j = tempK.top();
            j = out.firstChild[j];
            while ( out.nextSibling[j] != -1 ) j = out.nextSibling[j];
            out.nextSibling[j] = out.vNumber;
        }

        out.firstChild.push_back(-1); //the new node has no child yet
        out.nextSibling.push_back(-1); //the new node has no right sibling yet
        out.parent.push_back(tempK.top()); //the parent of the new node is the current node
        tempK.push(out.vNumber);
        out.vNumber++;
        if(out.vNumber >= CMR_MAX_TREE_NODES){
            throw std::domain_error("CMOrderedTreeMinerWrapper::addTree: tree has too many nodes.\n");
        }
    }
    return;
}

void CMOrderedTreeMinerWrapper::addTree(short treeDescription[], int length){
    TextTree tt;
    arrayToTextTree(treeDescription, length, ++current_tid, tt);
    database.push_back(tt);

    for(int i = 0 ; i < tt.vLabel.size() ; i++){
        short l = tt.vLabel[i];
        if(l < MIN_VERTEX){
            MIN_VERTEX = l;
        }else if(l > MAX_VERTEX){
            MAX_VERTEX = l;
        }
    }
    return;
}

void CMOrderedTreeMinerWrapper::reset(){
    database.clear();
    resetStats();
    currentPatternTree.initialSize();
}

void CMOrderedTreeMinerWrapper::mine(int support, vector<PatternTree> & maximal_out){

    currentPatternTree.initialSize();
    maximal_out.clear();
    vector<int> checked(CMR_MAX_TREE_NODES,0);
    vector<int> closed(CMR_MAX_TREE_NODES,0);
    vector<int> maximal(CMR_MAX_TREE_NODES,0);
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

void CMOrderedTreeMinerWrapper::treeToVector(TextTree & tt, vector<short> & v_out){
     v_out.clear();
     dfsVisit(0, tt, v_out);
     v_out.pop_back(); // remove the ending "-1"
     return;
}

void CMOrderedTreeMinerWrapper::timeShiftDatabaseHelper(const short vertex, const int level, const int tree_index){
    TextTree & ta = database.at(tree_index + level); // ta = tree ahead = tree for look ahead
    TextTree & tt = database.at(tree_index);         // tt = text tree
    tt.vLabel.at(vertex) = ta.vLabel.at(vertex);     // perform lookahead

    if(level <= 1){
        return; // dont need to do look ahead for bottom nodes
    }

    int child = tt.firstChild.at(vertex);
    if(child !=-1){
        timeShiftDatabaseHelper(child, level - 1, tree_index);
        int sib = tt.nextSibling.at(child);
        while(sib != -1){
            timeShiftDatabaseHelper(sib, level - 1, tree_index);
            sib = tt.nextSibling.at(sib);
        }
    }
    return;
}

void CMOrderedTreeMinerWrapper::timeShiftDatabase(int treeDepth){
    for(int i = 0 ; i < database.size() - treeDepth + 1; i++){
        timeShiftDatabaseHelper(0, treeDepth - 1, i);
    }

    // throw away the last trees
    for(int i = 0 ; i < treeDepth - 1; i++){
        if(database.size() > 0){
            database.pop_back();
        }
    }
    return;
}


int CMOrderedTreeMinerWrapper::treeMatchesHelper(const TextTree & parent_tree,
                                                  const TextTree & child_tree,
                                                  const short pt_start_node // parent tree start node
                                                  ){


    short cn_ct = 0; //cn_ct = current node child tree
    short cn_pt = pt_start_node; //cn_pt = current node parent tree
    while(true){
        if(parent_tree.vLabel.at(cn_pt) == child_tree.vLabel.at(cn_ct)){
            short fc_ct = child_tree.firstChild.at(cn_ct); // fc_ct =  first child of current node of child tree
            if(fc_ct != -1 ){
                //has child
                short fc_pt = parent_tree.firstChild.at(cn_pt); // first child of current node of parent tree
                if(fc_pt == -1){
                    return -1;
                }
                cn_ct = fc_ct; // set current node to
                cn_pt = fc_pt;
            }else while(true){
                int ct_next_sib = child_tree.nextSibling.at(cn_ct) ; // next sibling of the current node of the child tree
                if(ct_next_sib != -1){
                    cn_ct = ct_next_sib;
                    cn_pt = parent_tree.nextSibling.at(cn_pt);
                    if(cn_pt == -1){
                        return -1; //there was no next sibling
                    }
                    break; //break back to begining of outer while loop
                }else{
                    if(cn_ct == 0){
                        return cn_pt;
                    }
                    cn_ct = child_tree.parent.at(cn_ct);
                    cn_pt = parent_tree.parent.at(cn_pt);
                }
            }
        }else{
            cn_pt = parent_tree.nextSibling.at(cn_pt);
            if(cn_pt == -1){
                return -1; //current node of parent tree had no sibling
            }
        }
    }//end while
}

/**
 * @brief CMOrderedTreeMinerWrapper::isSubTreeOfHelper
 * In a recursive fashion, scan through the parent tree one node at a time.
 * At each parent node (vertex) detect if the child tree can fit there.
 * @param parent_tree - look for child_tree in this tree
 * @param child_tree - look for this tree in parent_tree
 * @param pt_vertex - Parent tree vertex. Position of child root node in parent tree to look.
 * @return true if child_tree is a subtree of the parent_tree at the pt_vertex position
 */
// TODO: Could be made faster by not checking the bottom "[ child tree depth - 1 ]" nodes.
// TODO: Remove bounds check on vectors ( remove .at method, just use [] operator).
// TODO: May not need this outer isSubTreeOfHelper method. May just need treeMatchesHelper.
int CMOrderedTreeMinerWrapper::isSubTreeOfHelper(const TextTree & parent_tree, const TextTree & child_tree, const short pt_vertex){
    if(pt_vertex >= parent_tree.vLabel.size()){
        return -1;
    }

    int match_pos; // match position - vertex of parent tree where root of child tree is a subtree
    match_pos = treeMatchesHelper(parent_tree, child_tree, pt_vertex);
    if(match_pos != -1){
        return match_pos;
    }

    int child = parent_tree.firstChild.at(pt_vertex); // child of the parent node
    // TODO: may be able to simplify by incrementing pt_vertex by one // instead of going to first child, should be the same thing if going depth first.
    if(child !=-1){
        match_pos = isSubTreeOfHelper(parent_tree, child_tree, child);
        if( match_pos != -1 ){
            return match_pos;
        }
        while( (child = parent_tree.nextSibling.at(child)) != -1){ // visit the rest of the children
            match_pos = isSubTreeOfHelper(parent_tree, child_tree, child);
            if(match_pos != -1){
                return match_pos;
            }
        }
    }
    return -1;
}

vector<int> CMOrderedTreeMinerWrapper::findSubtreeLocations(TextTree & haystack, TextTree & needle){
    vector<int> locations;
    for(int pos = 0; pos < haystack.vNumber; /*empty*/){
        int foundPos = isSubTreeOfHelper(haystack, needle, pos);
        if(foundPos == -1){
            pos++;
        }else{
            locations.push_back(foundPos);
            pos = foundPos + 1;
        }
    }
    return locations;
}
