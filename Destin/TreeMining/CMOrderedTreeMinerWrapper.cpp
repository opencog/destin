#include "CMOrderedTreeMinerWrapper.h"


void CMOrderedTreeMinerWrapper::addTree(short description[], int length){
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
        if(rhs.vNumber >= CMR_MAX_TREE_NODES){
            throw std::domain_error("CMOrderedTreeMinerWrapper::addTree: tree has too many nodes.\n");
        }
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

    for(int i = 0 ; i < treeDepth - 1; i++){
        if(database.size() > 0){
            database.pop_back();
        }
    }
    return;
}

bool CMOrderedTreeMinerWrapper::treeMatchesHelper(const TextTree & parent_tree,
                       const TextTree & child_tree,
                       const short pt_start_node,
                       const short ct_start_node){

    short cn_ct = ct_start_node; //cn_ct = current node child tree
    short cn_pt = pt_start_node; //cn_pt = current node parent tree

    while(true){
        if(parent_tree.vLabel.at(cn_pt) == child_tree.vLabel.at(cn_ct)){
            short fc_ct = child_tree.firstChild.at(cn_ct); // fc_ct =  first child of current node of child tree
            if(fc_ct != -1 ){
                //has child
                short fc_pt = parent_tree.firstChild.at(cn_pt); // first child of current node of parent tree
                if(fc_pt == -1){
                    return false;
                }
                cn_ct = fc_ct; // set current node to
                cn_pt = fc_pt;
            }else while(true){
                int ct_next_sib = child_tree.nextSibling.at(cn_ct) ;
                if(ct_next_sib != -1){
                    cn_ct = ct_next_sib;
                    cn_pt = parent_tree.nextSibling.at(cn_pt);
                    if(cn_pt == -1){
                        return false; //there was no next sibling
                    }
                    break; //break back to begining of outer while loop
                }else{
                    if(cn_ct == ct_start_node){
                        return true;
                    }
                    cn_ct = child_tree.parent.at(cn_ct);
                    cn_pt = parent_tree.parent.at(cn_pt);
                }
            }
        }else{
            cn_pt = parent_tree.nextSibling.at(cn_pt);
            if(cn_pt == -1){
                return false; //current node of parent tree had no sibling
            }
        }
    }//end while
}

bool CMOrderedTreeMinerWrapper::isSubTreeOfHelper(const TextTree & parent_tree, const TextTree & child_tree, const short pt_vertex){
    if(treeMatchesHelper(parent_tree, child_tree, pt_vertex, 0)){
        return true;
    }
    int child = parent_tree.firstChild.at(pt_vertex);
    if(child !=-1){
        if(isSubTreeOfHelper(parent_tree, child_tree, child)){
            return true;
        }
        while( (child = parent_tree.nextSibling.at(child)) != -1){
            if(isSubTreeOfHelper(parent_tree, child_tree, child)){
                return true;
            }
        }
    }
    return false;
}
