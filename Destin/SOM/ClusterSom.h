#ifndef CLUSTER_SOM_HPP
#define CLUSTER_SOM_HPP
#include <stdexcept>
#include <vector>

#include "ISom.h"
extern "C" {
    // define these prototypes instead of including "cluster/src/cluster.h" because
    // there's a name conflic with max macro and Node
double clusterdistance (int nrows, int ncolumns, double** data, int** mask,
    double weight[], int n1, int n2, int index1[], int index2[], char dist,
    char method, int transpose);

void somcluster (int nrows, int ncolumns, double** data, int** mask,
    const double weight[], int transpose, int nxnodes, int nynodes,
    double inittau, int niter, char dist, double*** celldata,
    int clusterid[][2]);
}



class ClusterSom : public ISom {
    std::vector< double * > trainData;
    const int rows, cols, dim;
    double*** celldata;
    float *** celldata_float;
    bool hasTrained;

    int * defaultMaskData;
    std::vector< int * > defaultMask;

    double * defaultWeight;
    char distMetric;
    double inittau;

    float _distance(double * data1, double * data2){

        double * data[2];
        data[0] = data1;
        data[1] = data2;

        int indicies1[1];
        indicies1[0] = 0;

        int indicies2[1];
        indicies2[0] = 1;

        return clusterdistance(2, dim, data, defaultMask.data(), defaultWeight, 1, 1, indicies1, indicies2, distMetric, 's', 0);
    }

public:

    ClusterSom(int rows, int cols, int dim):
        rows(rows),
        cols(cols),
        dim(dim),
        hasTrained(false),
        distMetric('e'),
        inittau(0.02)
    {
        if(rows <= 0 || cols <= 0 || dim <= 0){
            throw std::domain_error("rows, cols, and dim must be greater than 0");
        }
        // allocate grid units / cells / neurons / samples / clusters
        celldata = new double**[rows];
        celldata_float = new float**[rows];
        for(int r = 0 ; r < rows; r++){
            celldata[r] = new double*[cols];
            celldata_float[r] = new float*[cols];
            for(int c = 0 ; c < cols ; c++){
                celldata[r][c] = new double[dim];
                celldata_float[r][c] = new float[dim];
                for(int i = 0 ; i < dim ; i++){
                    celldata[r][c][i] = 0.0;
                    celldata_float[r][c][i] = 0.0;
                }
            }
        }

        defaultWeight = new double[dim];
        defaultMaskData = new int[dim];
        for(int i = 0 ; i < dim ; i++){
            defaultWeight[i] = 1.0;
            defaultMaskData[i] = 1;
        }
    }

    ~ClusterSom(){
        for(int i = 0 ; i < trainData.size() ; i++){
            delete [] trainData[i];
        }

        for(int r = 0; r < rows ; r++){
            for(int c = 0 ; c < cols; c++){
                delete [] celldata[r][c];
                delete [] celldata_float[r][c];
            }
            delete [] celldata[r];
            delete [] celldata_float[r];
        }
        delete [] celldata;
        delete [] celldata_float;

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
        if(trainData.size() == 0){
            throw runtime_error("can't train without adding data.\n");
        }
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
                   distMetric,
                   celldata,
                   NULL);


        // make a float copy of the cell data
        for(int r = 0 ;  r < rows ; r++){
            for(int c = 0; c < cols; c++){
                for(int i = 0; i < dim; i++){
                    celldata_float[r][c][i] = celldata[r][c][i];
                }
            }
        }

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
        distMetric = dist;
    }

    /** Find which SOM cell best matches the given data vector.
      * @return CvPoint with x=col y=row coordinates of the best matching cell.
      */
    CvPoint findBestMatchingUnit(float * data){
        if(!hasTrained){
            throw std::logic_error("Can't findBestMatchingUnit before calling train at least once.\n");
        }

        double distance;
        double minDist = 1e100;
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

    /** Add data for training
      * Makes a copy of the given data vector of lenght dim
      * to be used for training later
      *
      */
    void addTrainData(float * data){

        //convert the float into double data
        double * ddata = new double[dim]; //will be deleted/freed in the deconstructor
        for(int i = 0 ; i < dim ; i++){
            ddata[i] = data[i];
        }

        // The mask tells it that no data elements are missing
        defaultMask.push_back(defaultMaskData);
        //
        trainData.push_back(ddata);
    }


    float distance(float *data1, float *data2){
        double d1[dim];
        double d2[dim];
        std::copy(data1, data1 + dim, d1);
        std::copy(data2, data2 + dim, d2);
        return _distance(d1, d2);
    }

    float distance_coords(int r1, int c1, int r2, int c2){
        return _distance(celldata[r1][c1], celldata[r2][c2]);
    }

    void train_iterate(float * data){
        printf("ClusterSom::train_iterate not implemented\n");
    }

    float * getMapCell(int row, int col){
        return celldata_float[row][col];
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
