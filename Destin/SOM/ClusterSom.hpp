#ifndef CLUSTER_SOM_HPP
#define CLUSTER_SOM_HPP

#include <stdexcept>
#include <vector>
#include <limits>
#include "ISom.hpp"


//hack to get around Node name collision
#define Node ClusterNode
#include "cluster/src/cluster.h"
#define Node Node
//end hack


// Function pointer to various distance metrics
typedef double (*DistanceMetric)(
    int n,                  // The number of elements in a row or column. If transpose==0, then n is the number of columns; otherwise, n is the number of rows.
    double ** data1,        // The data array containing the first vector.
    double ** data2,        // The data array containing the second vector.
    int ** mask1,           // This array which elements in data1 are missing. If mask1[i][j]==0, then data1[i][j] is missing.
    int ** mask2,           // This array which elements in data2 are missing. If mask2[i][j]==0, then  data2[i][j] is missing.
    const double[] weight,  // The weights that are used to calculate the distance.
    int index1,             // Index of the first row or column.
    int index2,             // Index of the second row or column.
    int transpose           // If transpose==0, the distance between two rows in the matrix is calculated. Otherwise, the distance between two columns in the matrix is calculated.
    );

DistanceMetric setmetric(char d);


class ClusterSom : public ISom {
    std::vector< double * > trainData;
    const int rows, cols, dim;
    double*** celldata;
    bool hasTrained;

    int * defaultMaskData;
    std::vector< int * > defaultMask;

    double * defaultWeight;
    DistanceMetric metric;
    char distMetric;
    double inittau;

    float _distance(double * data1, double * data2){
        return metric(
            dim,
            &data1,
            &data2,
            defaultMask.data(),
            defaultMask.data(),
            defaultWeight,
            0,
            0,
            0);
    }

public:
    ClusterSom(int rows, int cols, int dim):
        rows(rows), cols(cols), dim(dim),
        hasTrained(false),
        distMetric('e'),
        metric(setmetric(distMetric),
        inittau(0.02)
    {
        if(rows <= 0 || cols <= 0 || dim <= 0){
            throw std::domain_error("rows, cols, and dim must be greater than 0");
        }
        // allocate grid units / cells / neurons / samples / clusters
        celldata = new double**[rows];

        for(int r = 0 ; r < rows; r++){
            celldata[r] = new double*[cols];
            for(int c = 0 ; c < cols ; c++){
                celldata[r][c] = new double[dim];
                for(int i = 0 ; i < dim ; i++){
                    celldata[r][c][i] = 0.0;
                }
            }
        }

        defaultWeight = new double[dim];
        defaultMaskData = new int[dim];
        for(int i = 0 ; i < dim ; i++){
            defaultWeight[i] = 1.0;
            defaultMaskData = 1;
        }
    }

    ~ClusterSom(){
        for(int i = 0 ; i < trainData.size() ; i++){
            delete [] trainData[i];
        }

        for(int r = 0; r < rows ; r++){
            for(int c = 0 ; c < cols; c++){
                delete [] celldata[r][c];
            }
            delete [] celldata[r];
        }
        delete [] celldata;

        delete [] defaultWeight;
        delete [] defaultMaskData;
    }

    /**
 int nrows ;
The number of rows in the data matrix, equal to the number of genes in the gene
expression experiment.
• int ncolumns ;
The number of columns in the data matrix, equal to the number of microarrays in the
gene expression experiment.
• double** data ;
The data array containing the gene expression data. Genes are stored row-wise, while
microarrays are stored column-wise. Dimension: [nrows ][ncolumns ].
• int** mask ;
This array shows which elements in the data array, if any, are missing.
mask [i][j]==0, then data [i][j] is missing. Dimension: [nrows ][ncolumns ].
If
• double weight [];
The weights that are used to calculate the distance. Dimension: [ncolumns ] if trans-
pose ==0; [nrows ] if transpose ==1.
• int transpose ;
This flag indicates whether row-wise (gene) or column-wise (microarray) clustering
is being performed. If transpose ==0, rows (genes) are being clustered. Otherwise,
columns (microarrays) are being clustered.
• int nxgrid ;
The number of cells horizontally in the rectangular topology containing the clusters.
• int nygrid ;
The number of cells vertically in the rectangular topology containing the clusters.
• double inittau ;
The initial value for the parameter τ that is used in the SOM algorithm. A typical
value for inittau is 0.02, which was used in Michael Eisen’s Cluster/TreeView program.
• int niter ;
The total number of iterations.
• char dist ;
Specifies which distance measure is used.
[Distance functions],

• double*** celldata ;
The data vectors of the clusters in the rectangular topology that were found by the
SOM algorithm. These correspond to the cluster centroids. The first dimension is the
horizontal position of the cluster in the rectangle, the second dimension is the vertical
position of the cluster in the rectangle, while the third dimension is the dimension along
the data vector. The somcluster routine does not allocate storage space for the celldata
array. Space should be allocated before calling somcluster. Alternatively, if celldata
is equal to NULL, the somcluster routine allocates space for celldata and frees it before
returning. In that case, somcluster does not return the data vectors of the clusters
that were found. Dimension: [nxgrid ][nygrid ][ncolumns ] if transpose ==0, or
[nxgrid ][nygrid ][nrows ] if transpose ==1.
Chapter 5: Self-Organizing Maps
23
• int clusterid [][2];
Specifies the cluster to which a gene or microarray was assigned, using two integers
to identify the horizontal and vertical position of a cell in the grid for each gene or
microarray. Gene or microarrays are assigned to clusters in the rectangular grid by
determining which cluster in the rectangular topology has the closest data vector.
Space for the clusterid argument should be allocated before calling somcluster. If
clusterid is NULL, the somcluster routine ignores this argument and does not return
the cluster assignments. Dimension: [nrows ][2] if transpose ==0; [ncolumns ][2] if
transpose ==1

      */

    void train(int n_iter ){
        int nrows = trainData.size();
        int ncolumns = dim;
        somcluster(nrows,
                   ncolumns,
                   trainData.data(),
                   defaultMask.data(),
                   defaultWeight,
                   0 /*transpose*/,
                   cols,
                   rows,
                   inittau,
                   n_iter,
                   dist,
                   celldata,
                   NULL);

        hasTrained = true;
    }


    /** Sets the distance metric identified by the given character
        case 'e': euclid
        case 'b': cityblock
        case 'c': correlation
        case 'a': acorrelation
        case 'u': ucorrelation
        case 'x': uacorrelation
        case 's': spearman
        case 'k': kendall
        any other: euclid
      */
    void setDistMetric(char dist){
        metric = setmetric(dist);
    }

    /** Find which SOM cell best matches the given data vector.
      * @return CvPoint with x=col y=row coordinates of the best matching cell.
      */
    CvPoint findBestMatchingUnit(float * data){
        if(!hasTrained){
            throw std::logic_error("Can't findBestMatchingUnit before calling train at least once.\n");
        }

        double distance;
        double minDist = std::numeric_limits<double>::max();
        CvPoint best;
        best.x = 0;
        best.y = 0;
        double ddata[dim];
        std::copy(data, data+dim, ddata); //copy float vector into a double vector
        for(int r = 0 ; r < rows ; r++){
            for(int c = 0 ; c < cols ; c++){
                distance = _distance(celldata[r][c], ddata);
                if(distance < minDist){
                    minDist = distance;
                    best.x = c;
                    best.y = r;
                }

            }
        }
        return best;
    }

    void addTrainData(float * data){

        //convert the float into double data
        double * ddata = new double[dim]; //will be deleted in the deconstructor
        for(int i = 0 ; i < dim ; i++){
            ddata[i] = data[i];
        }

        defaultMask.push_back(defaultMaskData);
        trainData.push_back(ddata);
    }


    float distance(float *data1, float *data2){
        double d1[dim];
        double d2[dim];
        std::copy(data1, data1 + dim, d1);
        std::copy(data2, data2 + dim, d2);
        return _distance(d1, d2);
    }

    void train_iterate(float * data){
        printf("ClusterSom::train_iterate not implemented\n");
    }

    float * getMapCell(int row, int col){

    }

    int cell_rows(){
        return rows;
    }

    int cell_cols(){
        return cols;
    }

    int cell_dim(){
        return dim;
    }

};

#endif
