/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
Header: PatternTree.h
		A PatternTree is a special TextTree with additional:
		1.	the vector for previousSibling, for quick locating the
			left sibling of a node
		2.	initialSize, to reserve enough space. Since there is only
			one PatternTree in the main program, the size doesn't matter
		3.	two functions for adding or removing the rightmost node
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef PatternTree_H
#define PatternTree_H

#include "CMRmisc.h"
#include "TextTree.h"

using namespace std;

struct PatternTree : public TextTree {
	PatternTree() : TextTree() {}
	PatternTree(int t) : TextTree(t) {}
	PatternTree(int t, short v) : TextTree(t,v) {}

	vector<short> previousSibling;

	void initialSize () 
	{
		tid = 0;
		vNumber = 0;
		vLabel.resize(1000,-1);
		firstChild.resize(1000,-1);
		nextSibling.resize(1000,-1);
		previousSibling.resize(1000,-1);
		parent.resize(1000,-1);
	}

	void deleteRightmost ()
	{
		if ( vNumber == 0 ) return;
		else if ( vNumber == 1 ) {
			vNumber = 0;
		}
		else {
			if ( firstChild[parent[vNumber-1]] == vNumber - 1 ) {
				firstChild[parent[vNumber-1]] = -1;
				vNumber--;
			}
			else {
				short k = firstChild[parent[vNumber-1]];
				while ( nextSibling[k] != vNumber - 1 ) k = nextSibling[k];
				nextSibling[k] = -1;
				vNumber--;
			}
		}
		return;
	}

	void addRightmost (short vertexLabel, short position)
	{
		if ( vNumber == 0 ) {
			vLabel[0] = vertexLabel;
			firstChild[0] = -1;
			nextSibling[0] = -1;
			previousSibling[0] = -1;
			parent[0] = -1;
			vNumber = 1;
		}
		else if ( position == vNumber ) { //a speical flag, means adding as the child of current rightmost node
			firstChild[vNumber-1] = vNumber;
			firstChild[vNumber] = -1;
			nextSibling[vNumber] = -1;
			previousSibling[vNumber] = -1;
			parent[vNumber] = vNumber - 1;
			vLabel[vNumber] = vertexLabel;
			vNumber++;
		}
		else { //assuming position is always on the rightmost path, and != 0
			firstChild[vNumber] = -1;
			nextSibling[vNumber] = -1;
			parent[vNumber] = parent[position];
			nextSibling[position] = vNumber;
			previousSibling[vNumber] = position;
			vLabel[vNumber] = vertexLabel;
			vNumber++;
		}
	}

};

#endif //PatternTree_H
