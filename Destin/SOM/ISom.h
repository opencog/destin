#ifndef ISOM_HPP
#define ISOM_HPP


using namespace std;

typedef void (*ItemToColorFunc)(float * item, int &r, int &g, int &b);

typedef unsigned int uint;

struct CvPoint;

/** An abstract interface for different flavors of SOM classes
  * Currently ClusterSom and BrlySom
  *
  */
class ISom {

public:
    virtual ~ISom(){}

    virtual CvPoint findBestMatchingUnit(float * data) = 0;

    virtual void addTrainData(float * data) = 0;

    virtual void train(int n_iters) = 0;

    virtual float distance(float * data1, float * data2) = 0;

    virtual float distance_coords(int r1, int c1, int r2, int c2) = 0;

    virtual float * getMapCell(int row, int col) = 0;

    virtual int cell_rows() = 0;

    virtual int cell_cols() = 0;

    virtual int cell_dim() = 0;

};

#endif // ISOM_HPP
