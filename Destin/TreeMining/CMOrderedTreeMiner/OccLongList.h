/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
Header: OccLongList.h
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef OccLongList_H
#define OccLongList_H

#include "CMRmisc.h"
#include "TextTree.h"
#include "PatternTree.h"
#include "OccList.h"

extern PatternTree currentPatternTree;

using namespace std;

struct OccLongList { //the occurrence list for the FreqTree, i.e., Asai's algorithm
	vector<pair<int,vector<short> > > occurrenceLong;
	int lastTid;
	int mySupport;

	OccLongList() : lastTid(-1), mySupport(0)
	{
	}

	void insert(int newTid, vector<short>& oldLocations, short newLocation);

	bool combineList(const OccLongList& mother, const OccList& newNodes);

	void explore(const vector<bool>& isFrequent, 
		const vector<TextTree>& database,
		const int& support,
		vector<int>& checked,
		vector<int>& closed,
		vector<int>& maximal);
};

#endif //OccLongList_H
