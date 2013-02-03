/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
Struct: TextTree

Description: Used to store a rooted ordered tree, i.e., 
a transaction in the database.
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "CMRmisc.h"
#include "TextTree.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
operator>>()

Decription: read in a rooted ordered tree in Zaki's format
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
istream& operator>>(istream& in, TextTree& rhs)
{
	int total;
	short temp;
	stack<short> tempK;

	in >> rhs.tid;
	if ( in.eof() ) return in; //the end of file, no more trees to read
	in >> rhs.tid; //read tid twice

	in >> total;
	in >> temp; //read in the root label

	rhs.vLabel.push_back(temp);
	rhs.firstChild.push_back(-1); //temporarily, the root has no child
	rhs.nextSibling.push_back(-1); //the root has no sibling
	rhs.parent.push_back(-1); //the root has no parent
	rhs.vNumber = 1;

	tempK.push(0); //the index of the root

	for ( short i = 1; i < total; i++ ) {
		in >> temp;
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
	return in;
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
dfsVisit(const short, const TextTree& rhs, vector<short>& zakiCode)

Decription: a helper function for depth-first-visit
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void dfsVisit(const short current, const TextTree& rhs, vector<short>& zakiCode)
{
	zakiCode.push_back(rhs.vLabel[current]);
	short i = rhs.firstChild[current];
	while ( i != -1 )
	{
		dfsVisit(i,rhs,zakiCode);
		i = rhs.nextSibling[i];
	}
	zakiCode.push_back(-1);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
operator<<()

Decription: write out a rooted ordered tree in Zaki's format
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
ostream& operator<<(ostream& out, const TextTree& rhs)
{
	out << rhs.tid << ' ' << rhs.tid << ' ';

	vector<short> zakiCode;
	dfsVisit(0, rhs, zakiCode);
	out << zakiCode.size() - 1 << ' '; //Zaki's code has one less "-1" than Luccio's
	for ( short i = 0; i < zakiCode.size() - 1; i++ )
		out << zakiCode[i] << ' ';
	out << endl;
	return out;
}  
