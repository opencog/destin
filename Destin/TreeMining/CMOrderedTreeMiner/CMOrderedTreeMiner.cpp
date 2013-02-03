// CMOrderedTreeMiner.cpp : Defines the entry point for the console application.
#include "CMRmisc.h"
#include "TextTree.h"
#include "PatternTree.h"
#include "OccList.h"
#include "OccLongList.h"

short MIN_VERTEX = 30001;
short MAX_VERTEX = -1; //therefore, the range for valid node label is 0--30000

short currentVertexNumber;

PatternTree currentPatternTree;

int main(int argc, char* argv[])
{
	if ( argc != 4 )
	{	
		cout << "Usage: CMOrderedTreeMiner support input_file output_file" << endl;
		exit (1);
	}

	int support;
	istringstream iss(argv[1]);
	iss >> support;
	if(!iss)
	{
		cerr << "invalid argument, not an integer value" << endl;
		exit (1);
	}

	vector<int> frequency(1000,0); //assuming the max frequent tree size is 1000
	vector<int> checked(1000,0);
	vector<int> closed(1000,0);
	vector<int> maximal(1000,0);

	currentPatternTree.initialSize();
	time_t start_time;
	time_t stop_time;

	/******************************************************************
	step1: read in the database, and find the MIN_VERTEX and MAX_VERTEX
	******************************************************************/
	string inputFile = argv[2];
	string outputFile = argv[3];

	ofstream outFile(outputFile.c_str());
	if(!outFile) {
		cerr << "cannot open OUTPUT file!" << endl;
		exit(1);
	}

	ifstream inFile(inputFile.c_str());
	if(!inFile) {
		cerr << "cannot open INPUT file!" << endl;
		exit(1);
	}

	vector<TextTree> database;
	int myTid = 0;
	while ( !inFile.eof() ) {
		TextTree tt;
		inFile >> tt;
		if ( !inFile.eof() ) {
			tt.tid = myTid++;
			for ( short i = 0; i < tt.vNumber; i++ ) {
				if ( tt.vLabel[i] < MIN_VERTEX ) MIN_VERTEX = tt.vLabel[i];
				if ( tt.vLabel[i] > MAX_VERTEX ) MAX_VERTEX = tt.vLabel[i];
			}
			database.push_back(tt);
		}
	}
	inFile.close();

	/******************************************************************
	step2.1: scan the database once, find frequent node labels
	******************************************************************/
	vector<bool> isFrequent(MAX_VERTEX - MIN_VERTEX + 1, false);
	map<short,int> count;
	map<short,int>::iterator pos;

	start_time = time(0);

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
			pos2->second.explore(isFrequent,database,support,checked,closed,maximal);
			currentPatternTree.deleteRightmost();
		}
	}

	stop_time = time(0);

	/******************************************************************
	step2.4: output the results 
	******************************************************************/
	for ( short j = 0; j < 1000; j++ ) {
		if ( checked[j] > 0 ) {
			outFile << "number of checked " << j << " trees: " << checked[j] << endl;
		}
	}
	outFile << endl << "************************" << endl;
	for ( short j = 0; j < 1000; j++ ) {
		if ( closed[j] > 0 ) {
			outFile << "number of closed " << j << " trees: " << closed[j] << endl;
		}
	}
	outFile << endl << "************************" << endl;
	for ( short j = 0; j < 1000; j++ ) {
		if ( maximal[j] > 0 ) {
			outFile << "number of maximal " << j << " trees: " << maximal[j] << endl;
		}
	}

	outFile << endl;
	outFile << "Total Running Time: " << difftime(stop_time, start_time) << endl;

	outFile.close();


	return 0;
}
