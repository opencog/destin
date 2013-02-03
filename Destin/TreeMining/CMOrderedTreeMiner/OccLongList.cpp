/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
Class: OccLongList

Description: Used to store the occurrence list of a rooted 
ordered pattern tree, for CMOrderedTreeMiner algorithms
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "OccLongList.h"

void OccLongList::insert(int newTid, vector<short>& oldLocations, short newLocation)
{
	occurrenceLong.push_back(make_pair(newTid,oldLocations));
	occurrenceLong.back().second.push_back(newLocation);
	if ( lastTid != newTid ) { //if comes from a new transaction
		lastTid = newTid;
		mySupport++;
	}
}

bool OccLongList::combineList(const OccLongList& mother, const OccList& newNodes)
{
	occurrenceLong.clear();
	int nodeId = 0;
	int motherId = 0;

	//check if the extension is a occurrenceMatch
	bool occurrenceMatch = true;
	int newNodeId = 0;
	motherId = -1;
	while ( newNodeId < newNodes.occurrence.size() ) {
		if ( newNodes.occurrence[newNodeId].first == (motherId+1) )
			motherId++;
		else if ( newNodes.occurrence[newNodeId].first != motherId ) {
			occurrenceMatch = false;
			break;
		}
		newNodeId++;
	}

	if ( occurrenceMatch ) { //have to pass the last test, i.e., if the last one
		//in newNodes has the greatest motherId
		if ( newNodes.occurrence[newNodeId-1].first != (mother.occurrenceLong.size()-1) )
			occurrenceMatch = false;
	}

	nodeId = 0;
	motherId = 0;
	mySupport = 0;
	int tempTid = -1;
	while ( nodeId < newNodes.occurrence.size() ) {
		//first, find the correct mother
		while ( motherId != newNodes.occurrence[nodeId].first ) motherId++;
		occurrenceLong.push_back(mother.occurrenceLong[motherId]);
		occurrenceLong.back().second.push_back(newNodes.occurrence[nodeId].second);
		if ( tempTid != mother.occurrenceLong[motherId].first ) {
			tempTid = mother.occurrenceLong[motherId].first;
			mySupport++;
		}
		while ( (nodeId+1) < newNodes.occurrence.size() 
			&& (newNodes.occurrence[nodeId].first == newNodes.occurrence[nodeId+1].first) ) {
				nodeId++;
				occurrenceLong.push_back(mother.occurrenceLong[motherId]);
				occurrenceLong.back().second.push_back(newNodes.occurrence[nodeId].second);
			}
			nodeId++;
	}
	return occurrenceMatch;
}


void OccLongList::explore(const vector<bool>& isFrequent, 
						  const vector<TextTree>& database,
						  const int& support,
						  vector<int>& checked,
						  vector<int>& closed,
						  vector<int>& maximal)
{
	
	//for debug
	//cout << currentPatternTree;
	//cout << "support of current pattern tree is: " << mySupport << endl;
	
	short tempV = currentPatternTree.vNumber;
	checked[tempV]++; //update the number of checked trees

	vector<bool> rightPath(tempV,false);
	short temp = tempV - 1;
	while ( temp != -1 ) {
		rightPath[temp] = true;
		temp = currentPatternTree.parent[temp];
	}

	set<pair<short, short> > occurrenceMatch; //(short,short) = (position,label)
	set<pair<short, short> >::iterator pos2;

	//step 1, do the occurrence match check
	int currentIndex; //current index of the occurrenceLong
	int myTid;
	int myLocation;

	//step 1.1, construct the base for occurrenceMatch
	//look at the root (0-th vertex) of the pattern tree
	currentIndex = 0;
	myTid = occurrenceLong[currentIndex].first;
	myLocation = occurrenceLong[currentIndex].second[0];
	if ( myLocation != 0 ) { //the rooted of the pattern tree is not the root of the transaction
		occurrenceMatch.insert(make_pair(0,database[myTid].vLabel[database[myTid].parent[myLocation]]));
	}

	//step 1.2, construct the base for occurrenceMatch
	//look at the "left","below", and "right" of all other nodes (not below the rightmost node)
	for ( short i = 1; i < tempV; i++ ) {
		myLocation = occurrenceLong[currentIndex].second[i];
		short j;

		//record left occurrences
		if ( currentPatternTree.previousSibling[i] == -1 ) //I have no left sibling
			j = database[myTid].firstChild[database[myTid].parent[myLocation]];
		else {
			j = occurrenceLong[currentIndex].second[currentPatternTree.previousSibling[i]];
			j = database[myTid].nextSibling[j];
		}
		while ( j != myLocation ) {
			occurrenceMatch.insert(make_pair(i,database[myTid].vLabel[j]));
			j = database[myTid].nextSibling[j];
		}

		//record below occurrences
		if ( currentPatternTree.firstChild[i] == -1 && i != (tempV-1) ) { 
			//I have no children
			//and I am not the rightmost node
			j = database[myTid].firstChild[myLocation];
			while ( j != -1 ) {
				occurrenceMatch.insert(make_pair(i+tempV,database[myTid].vLabel[j]));
				j = database[myTid].nextSibling[j];
			}
		}

		//record right occurrences
		if ( currentPatternTree.nextSibling[i] == -1 && !rightPath[i] ) { 
			//I have no right sibling
			//and I am not on the rightmost path
			j = database[myTid].nextSibling[myLocation];
			while ( j != -1 ) {
				occurrenceMatch.insert(make_pair(i+2*tempV,database[myTid].vLabel[j]));
				j = database[myTid].nextSibling[j];
			}
		}

	}

	//step 1.3, check the occurrence match
	currentIndex++;
	while ( occurrenceMatch.size() != 0 && currentIndex < occurrenceLong.size() ) {
		myTid = occurrenceLong[currentIndex].first;
		pos2 = occurrenceMatch.begin();
		while ( pos2 != occurrenceMatch.end() ) {
			short j;

			if ( pos2->first == 0 ) { //root occurrence
				myLocation = occurrenceLong[currentIndex].second[pos2->first];
				bool isFound = false;

				if ( pos2->second == database[myTid].vLabel[database[myTid].parent[myLocation]] )
					isFound = true;

				if ( !isFound )
					occurrenceMatch.erase(pos2++);
				else
					pos2++;
			}
			else if ( pos2->first < tempV ) { //a left occurrence
				myLocation = occurrenceLong[currentIndex].second[pos2->first];
				bool isFound = false;

				if ( currentPatternTree.previousSibling[pos2->first] == -1 ) //I have no left sibling
					j = database[myTid].firstChild[database[myTid].parent[myLocation]];
				else {
					j = occurrenceLong[currentIndex].second[currentPatternTree.previousSibling[pos2->first]];
					j = database[myTid].nextSibling[j];
				}

				while ( j != myLocation ) {
					if ( database[myTid].vLabel[j] == pos2->second ) { //found it!
						isFound = true;
						break;
					}
					j = database[myTid].nextSibling[j];
				}

				if ( !isFound )
					occurrenceMatch.erase(pos2++);
				else
					pos2++;
			}
			else if ( tempV < pos2->first && pos2->first < 2*tempV ) { //a below occurrence
				myLocation = occurrenceLong[currentIndex].second[pos2->first - tempV];
				bool isFound = false;

				j = database[myTid].firstChild[myLocation];
				while ( j != -1 ) {
					if ( database[myTid].vLabel[j] == pos2->second ) { //found it!
						isFound = true;
						break;
					}
					j = database[myTid].nextSibling[j];
				}

				if ( !isFound )
					occurrenceMatch.erase(pos2++);
				else
					pos2++;
			}
			else { //a right occurrence
				myLocation = occurrenceLong[currentIndex].second[pos2->first - 2*tempV];
				bool isFound = false;

				j = database[myTid].nextSibling[myLocation];
				while ( j != -1 ) {
					if ( database[myTid].vLabel[j] == pos2->second ) { //found it!
						isFound = true;
						break;
					}
					j = database[myTid].nextSibling[j];
				}

				if ( !isFound )
					occurrenceMatch.erase(pos2++);
				else
					pos2++;
			}
		}
		currentIndex++;
	}

	//step 1.4 prune, if there is occurrence match
	if ( occurrenceMatch.size() != 0 ) return;

	//step 2, do transaction match checking
	bool isClosed = true; //innocent until proven guilty
	bool isMaximal = true;
	set<pair<short, short> > transactionMatch;

	currentIndex = 0;
	int startIndex = 0; //the lower bound of a range
	int stopIndex = 0; //the upper bound of a range

	myTid = occurrenceLong[0].first;
	while ( (stopIndex+1) < occurrenceLong.size() 
		&& myTid == occurrenceLong[stopIndex+1].first )
		stopIndex++;

	for ( currentIndex = startIndex; currentIndex <= stopIndex; currentIndex++ ) {
		myLocation = occurrenceLong[currentIndex].second[0];

		//step 2.1, construct the base for transactionMatch
		//look at the root (0-th vertex) of the pattern tree
		if ( myLocation != 0 ) { //the rooted of the pattern tree is not the root of the transaction
			transactionMatch.insert(make_pair(0,database[myTid].vLabel[database[myTid].parent[myLocation]]));
		}

		//step 2.2, construct the base for transactionMatch
		//look at the "left", "below", and "right" of all other nodes (not below the rightmost node)
		for ( short i = 1; i < tempV; i++ ) {
			myLocation = occurrenceLong[currentIndex].second[i];
			short j;

			//record left occurrences
			if ( currentPatternTree.previousSibling[i] == -1 ) //I have no left sibling
				j = database[myTid].firstChild[database[myTid].parent[myLocation]];
			else {
				j = occurrenceLong[currentIndex].second[currentPatternTree.previousSibling[i]];
				j = database[myTid].nextSibling[j];
			}
			while ( j != myLocation ) {
				transactionMatch.insert(make_pair(i,database[myTid].vLabel[j]));
				j = database[myTid].nextSibling[j];
			}

			//record below occurrences
			if ( currentPatternTree.firstChild[i] == -1 && i != (tempV-1) ) { 
				//I have no children
				//and I am not the rightmost node
				j = database[myTid].firstChild[myLocation];
				while ( j != -1 ) {
					transactionMatch.insert(make_pair(i+tempV,database[myTid].vLabel[j]));
					j = database[myTid].nextSibling[j];
				}
			}

			//record right occurrences
			if ( currentPatternTree.nextSibling[i] == -1 && !rightPath[i] ) { 
				//I have no right sibling
				//and I am not on the rightmost path
				j = database[myTid].nextSibling[myLocation];
				while ( j != -1 ) {
					occurrenceMatch.insert(make_pair(i+2*tempV,database[myTid].vLabel[j]));
					j = database[myTid].nextSibling[j];
				}
			}

		}
	}

	//step 2.3, check the transaction match
	startIndex = stopIndex + 1;
	while ( transactionMatch.size() != 0 && startIndex < occurrenceLong.size() ) {
		stopIndex = startIndex;
		myTid = occurrenceLong[startIndex].first;

		while ( (stopIndex+1) < occurrenceLong.size() 
			&& myTid == occurrenceLong[stopIndex+1].first )
			stopIndex++;

		pos2 = transactionMatch.begin();
		while ( pos2 != transactionMatch.end() ) {
			short j;

			if ( pos2->first == 0 ) { //root occurrence
				bool isFound = false;
				currentIndex = startIndex;

				while ( !isFound && currentIndex <= stopIndex ) {
					myLocation = occurrenceLong[currentIndex].second[pos2->first];

					if ( pos2->second == database[myTid].vLabel[database[myTid].parent[myLocation]] )
						isFound = true;
					
					currentIndex++;
				}

				if ( !isFound )
					transactionMatch.erase(pos2++);
				else
					pos2++;
			}
			else if ( pos2->first < tempV ) { //a left occurrence
				bool isFound = false;
				currentIndex = startIndex;

				while ( !isFound && currentIndex <= stopIndex ) {
					myLocation = occurrenceLong[currentIndex].second[pos2->first];
					if ( currentPatternTree.previousSibling[pos2->first] == -1 ) //I have no left sibling
						j = database[myTid].firstChild[database[myTid].parent[myLocation]];
					else {
						j = occurrenceLong[currentIndex].second[currentPatternTree.previousSibling[pos2->first]];
						j = database[myTid].nextSibling[j];
					}

					while ( j != myLocation ) {
						if ( database[myTid].vLabel[j] == pos2->second ) { //found it!
							isFound = true;
							break;
						}
						j = database[myTid].nextSibling[j];
					}
					currentIndex++;
				}

				if ( !isFound )
					transactionMatch.erase(pos2++);
				else
					pos2++;
			}
			else if ( tempV < pos2->first && pos2->first < 2*tempV ) { //a below occurrence
				bool isFound = false;
				currentIndex = startIndex;

				while ( !isFound && currentIndex <= stopIndex ) {
					myLocation = occurrenceLong[currentIndex].second[pos2->first - tempV];

					j = database[myTid].firstChild[myLocation];
					while ( j != -1 ) {
						if ( database[myTid].vLabel[j] == pos2->second ) { //found it!
							isFound = true;
							break;
						}
						j = database[myTid].nextSibling[j];
					}
					currentIndex++;
				}

				if ( !isFound )
					transactionMatch.erase(pos2++);
				else
					pos2++;
			}
			else { //a right occurrence
				bool isFound = false;
				currentIndex = startIndex;

				while ( !isFound && currentIndex <= stopIndex ) {
					myLocation = occurrenceLong[currentIndex].second[pos2->first - 2*tempV];

					j = database[myTid].nextSibling[myLocation];
					while ( j != -1 ) {
						if ( database[myTid].vLabel[j] == pos2->second ) { //found it!
							isFound = true;
							break;
						}
						j = database[myTid].nextSibling[j];
					}
				}
			}
		}
		
		startIndex = stopIndex + 1;
	}

	//step 2.4 not closed or maximal, if there is transaction match
	if ( transactionMatch.size() != 0 ) {
		isClosed = false;
		isMaximal = false;
	}

	//step 3, explore all the right expansions
	bool isRightOccurrenceMatch = false;

	//step 3.1, explore the children of the rightmost node 
	map<short,OccList> potentialChildren;
	map<short,OccList>::iterator pos;
	for ( int i = 0; i < occurrenceLong.size(); i++ ) {
		int myTid = occurrenceLong[i].first;
		short myLocation = occurrenceLong[i].second[tempV-1];
		short k = database[myTid].firstChild[myLocation];
		//here, redundancy must be recorded also.
		while ( k != -1 ) {
			if ( isFrequent[database[myTid].vLabel[k] - MIN_VERTEX] == true ) {
				potentialChildren[database[myTid].vLabel[k] - MIN_VERTEX].insert(myTid,k,i);
			}
			k = database[myTid].nextSibling[k];
		}
	}

	for ( pos = potentialChildren.begin(); pos != potentialChildren.end(); ++pos ) {
		if ( pos->second.mySupport >= support ) { //a frequent extension!
			if ( pos->second.mySupport == mySupport )
				isClosed = false;
			isMaximal = false;
			currentPatternTree.addRightmost(pos->first + MIN_VERTEX,currentPatternTree.vNumber);
			OccLongList newLongList;
			if ( newLongList.combineList(*this,pos->second) ) {
				isRightOccurrenceMatch = true;
				isClosed = false;
				isMaximal = false;
			}
			newLongList.explore(isFrequent,database,support,checked,closed,maximal);
			currentPatternTree.deleteRightmost();
		}
	}

	//step 3.2, explore the right siblings of all the ancestors of the rightmost node
	short tempNode = currentPatternTree.vNumber - 1;
	while ( tempNode != 0 && !isRightOccurrenceMatch ) {
		potentialChildren.clear();
		for ( int i = 0; i < occurrenceLong.size(); i++ ) {
			int myTid = occurrenceLong[i].first;
			short myLocation = occurrenceLong[i].second[tempNode];
			short k = database[myTid].nextSibling[myLocation];
			while ( k != -1 ) {
				if ( isFrequent[database[myTid].vLabel[k] - MIN_VERTEX] == true ) {
					potentialChildren[database[myTid].vLabel[k] - MIN_VERTEX].insert(myTid,k,i);
				}
				k = database[myTid].nextSibling[k];
			}
		}

		for ( pos = potentialChildren.begin(); pos != potentialChildren.end(); ++pos ) {
			if ( pos->second.mySupport >= support ) { //a frequent extension!
				if ( pos->second.mySupport == mySupport )
					isClosed = false;
				isMaximal = false;
				currentPatternTree.addRightmost(pos->first + MIN_VERTEX,tempNode);
				OccLongList newLongList;
				if ( newLongList.combineList(*this,pos->second) ) {
					isRightOccurrenceMatch = true;
					isClosed = false;
					isMaximal = false;
				}
				newLongList.explore(isFrequent,database,support,checked,closed,maximal);
				currentPatternTree.deleteRightmost();
			}
		}
		tempNode = currentPatternTree.parent[tempNode];
	}

	//step 4, the worst case, need to check left blanket to see if maximal
	map<pair<short,short>, pair<int,int> > potentialBlanket; 
	//(short,short) = (position,label)
	//(int, int) = (lastTid,count)

	map<pair<short,short>, pair<int,int> >::iterator pos3;

	currentIndex = 0;
	while ( isMaximal && currentIndex < occurrenceLong.size() ) {
		myTid = occurrenceLong[currentIndex].first;

		//step 4.1, look at the root (0-th vertex) of the pattern tree
		myLocation = occurrenceLong[currentIndex].second[0];
		if ( myLocation != 0 ) { //the rooted of the pattern tree is not the root of the transaction
			pos3 = potentialBlanket.find(make_pair(0,database[myTid].vLabel[database[myTid].parent[myLocation]]));
			if ( pos3 != potentialBlanket.end() ) {
				if ( pos3->second.first != myTid ) {
					pos3->second.first = myTid;
					pos3->second.second++;
					if ( pos3->second.second >= support ) isMaximal = false;
				}
			}
			else {
				potentialBlanket.insert(make_pair(make_pair(0,database[myTid].vLabel[database[myTid].parent[myLocation]]),make_pair(myTid,1)));
			}
		}

		//step 4.2, look at the "left", "below", and "right" of all other nodes (not below the rightmost node)
		for ( short i = 1; i < tempV && isMaximal; i++ ) {
			myLocation = occurrenceLong[currentIndex].second[i];
			short j;

			//record left occurrences
			if ( currentPatternTree.previousSibling[i] == -1 ) //I have no left sibling
				j = database[myTid].firstChild[database[myTid].parent[myLocation]];
			else {
				j = occurrenceLong[currentIndex].second[currentPatternTree.previousSibling[i]];
				j = database[myTid].nextSibling[j];
			}
			while ( j != myLocation ) {
				pos3 = potentialBlanket.find(make_pair(i,database[myTid].vLabel[j]));
				if ( pos3 != potentialBlanket.end() ) {
					if ( pos3->second.first != myTid ) {
						pos3->second.first = myTid;
						pos3->second.second++;
						if ( pos3->second.second >= support ) isMaximal = false;
					}
				}
				else {
					potentialBlanket.insert(make_pair(make_pair(i,database[myTid].vLabel[j]),make_pair(myTid,1)));
				}
				j = database[myTid].nextSibling[j];
			}

			//record below occurrences
			if ( isMaximal && (currentPatternTree.firstChild[i] == -1 && i != (tempV-1)) ) { 
				//I have no children and I am not the rightmost node
				j = database[myTid].firstChild[myLocation];
				while ( j != -1 ) {
					pos3 = potentialBlanket.find(make_pair(i+tempV,database[myTid].vLabel[j]));
					if ( pos3 != potentialBlanket.end() ) {
						if ( pos3->second.first != myTid ) {
							pos3->second.first = myTid;
							pos3->second.second++;
							if ( pos3->second.second >= support ) isMaximal = false;
						}
					}
					else {
						potentialBlanket.insert(make_pair(make_pair(i+tempV,database[myTid].vLabel[j]),make_pair(myTid,1)));
					}
					j = database[myTid].nextSibling[j];
				}
			}

			//record right occurrence
			if ( isMaximal && (currentPatternTree.nextSibling[i] == -1 && !rightPath[i]) ) { 
				//I have no right sibling and I am not on the right path
				j = database[myTid].nextSibling[myLocation];
				while ( j != -1 ) {
					pos3 = potentialBlanket.find(make_pair(i+2*tempV,database[myTid].vLabel[j]));
					if ( pos3 != potentialBlanket.end() ) {
						if ( pos3->second.first != myTid ) {
							pos3->second.first = myTid;
							pos3->second.second++;
							if ( pos3->second.second >= support ) isMaximal = false;
						}
					}
					else {
						potentialBlanket.insert(make_pair(make_pair(i+2*tempV,database[myTid].vLabel[j]),make_pair(myTid,1)));
					}
					j = database[myTid].nextSibling[j];
				}
			}

		}

		if ( !isMaximal ) break;
		currentIndex++;
	}

	if ( isClosed ) {
		closed[tempV]++;
		//cout << "closed  " << tempV << ":" << currentPatternTree << " support=" << mySupport << endl;
	}
	if ( isMaximal ) {
		maximal[tempV]++;
		//cout << "maximal " << tempV << ":" << currentPatternTree << " support=" << mySupport << endl;
	}
}
