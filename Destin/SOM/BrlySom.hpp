#ifndef BrlySom_hpp
#define BrlySom_hpp

#include <stdio.h>
#include <opencv/cv.h>

#include "ISom.hpp"
#include "brly_som.hpp"


class BrlySom : public ISom {

    brly_som::Som<float> som;

    typedef const brly_som::Som<float>::Mat SomMat;
    typedef brly_som::Som<float>::Elem Sample;

    uint rows, cols, vector_dim;

public:

    BrlySom(uint map_rows, uint map_cols, uint vector_dim)
        :som(map_rows, map_cols, vector_dim),
          rows(map_rows), cols(map_cols), vector_dim(vector_dim){ }

    void train_iterate(float * data){
        som.next(data);
    }


    CvPoint findBestMatchingUnit(
        float *  // data to search with
        );

    float distance(float *cell1, float *cell2){
        return som.distance(cell1, cell2);
    }

    float distance_coords(int r1, int c1, int r2, int c2){
        float * cell1 = getMapCell(r1, r2);

        float * cell2 = getMapCell(r2, c2);
        return distance(cell1, cell2);
    }

    int cell_rows(){
        return rows;
    }

    int cell_cols(){
        return cols;
    }
    int cell_dim(){
        return vector_dim;
    }

    float * getMapCell(int row, int col){
        return &(som.data()[row][col][0]);
    }

    bool saveSom(
        string  //where to save the file
        );

    bool loadSom(
        string  //file to load
        );
};

#endif
