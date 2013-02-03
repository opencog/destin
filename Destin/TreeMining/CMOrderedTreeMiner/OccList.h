/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
Header: OccList.h
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef OccList_H
#define OccList_H

#include "CMRmisc.h"
#include "TextTree.h"

using namespace std;

struct OccList { //the occurrence list for the FREQT, i.e., Asai's algorithm
	vector<pair<int,short> > occurrence;
	int lastTid;
	int mySupport;

	OccList() : lastTid(-1), mySupport(0)
	{
	}

	void insert(int& newTid, const short& newLocation);

	void insert(int& newTid, const short& newLocation, int& motherId);

	void explore(const vector<bool>& isFrequent, 
		const vector<TextTree>& database,
		const int& support,
		vector<int>& frequency,
		const short level);
};

#endif //OccList_H
