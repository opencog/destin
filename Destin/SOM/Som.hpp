#ifndef SOM_HPP
#define SOM_HPP

#include "ISom.hpp"

#include "brly_som.hpp"

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

class Som : public ISom {

    brly_som::Som<float> som;

    cv::Mat sim_map;            //simularity map
    cv::Mat resized_sim_map;

    uint rows;
    uint cols;
    uint vector_dim;

    typedef const brly_som::Som<float>::Mat SomMat;
    typedef brly_som::Som<float>::Elem Sample;

    void calcSimularityMap(uint nh_width = 2){
        SomMat &m = som.data();

        unsigned int r,//row
                     c;//col
        int nr, //neigbor row
            nc; //neighbor col (can be negative if near the border)


        const Sample * neighbor;
        const Sample * origin;

        double distTotal;
        double maxDistance = -1.0;

        // For each point on the SOM, calcuate its simularity to all its
        // neighbors and use the total simularity value as a grayscale pixel
        // for the grayscale simularity map.
        float * sim_data = (float *)sim_map.data;//simularity map raw data
        for(r = nh_width ; r < rows - nh_width; r++){
            for(c = nh_width; c < cols - nh_width; c++){
                origin = &m[r][c];
                distTotal = 0;
                for(nr = r - nh_width ; nr < r + nh_width ; nr++){
                    if(nr < 0 || nr >= rows){ continue; }

                    for(nc = c - nh_width ; nc < c + nh_width ; nc++){
                        if(nc < 0 || nc >= cols){ continue; }
                        neighbor = &m[nr][nc];
                        distTotal += som.distance((Sample &)*origin, (Sample &)*neighbor);
                    }
                }
                sim_data[r * rows + c] = distTotal;
                //compute max distance for normalization
                if(distTotal > maxDistance){
                    maxDistance = distTotal;
                }
            }
        }

        int pixels = rows * cols;
        // normalize values from 0 to 1.0
        // so the image can be displayed as greyscale
        for(int i = 0; i < pixels; i++){
            sim_data[i] /= maxDistance;
        }
    }

public:

    Som(int map_rows, int map_cols, int vector_dim)
        :rows(map_rows),
          cols(map_cols),
          vector_dim(vector_dim),
          som(map_rows, map_cols, vector_dim),
          sim_map(map_rows, map_cols, CV_32FC1)
    {


    }

    void train_iterate(float * data){
        som.next(data);
    }


    void showSimularityMap(string windowName, uint nh_width = 2, int window_width = 512, int window_height = 512){
        calcSimularityMap(nh_width);
        cv::resize(sim_map, resized_sim_map, cv::Size(window_width, window_height));
        cv::imshow(windowName, resized_sim_map);
        cv::waitKey(5);
    }

    void showMap(string windowName){

    }

    void setItemToColorFuncPointer(ItemToColorFunc function){

    }

    bool saveSom(string filename){
        FILE * f;
        f = fopen(filename.c_str(), "w");
        if(f == NULL){
            fprintf(stderr, "Could not open file %s for writing.\n", filename.c_str());
            return false;
        }

        size_t wc  = 0; //write count

        wc += fwrite(&rows, sizeof(uint), 1, f );
        wc += fwrite(&cols, sizeof(uint), 1, f );
        wc += fwrite(&vector_dim, sizeof(uint), 1, f );
        uint t = som.getT();
        wc += fwrite(&t, sizeof(uint), 1, f);

        for(int r = 0 ; r < rows ; r++){
            for(int c = 0 ; c < cols ; c++){
                wc += fwrite(&(som.data()[r][c][0]), sizeof(float), vector_dim, f);
            }
        }

        fclose(f);

        size_t should_write = 4 + rows * cols  * vector_dim;

        if(wc != should_write){
            fprintf(stderr, "Trouble saving file %s\n", filename.c_str());
            fprintf(stderr, "Expected %lu, but saved %lu\n", should_write, wc);
            return false;
        }
        return true;

    }

    bool loadSom(string filename){
        FILE * f;
        f = fopen(filename.c_str(), "r");
        if(f == NULL){
            fprintf(stderr, "Could not open file %s for reading.\n", filename.c_str());
            return false;
        }

        size_t rc  = 0; //read count

        rc += fread(&rows, sizeof(uint), 1, f );
        rc += fread(&cols, sizeof(uint), 1, f );
        rc += fread(&vector_dim, sizeof(uint), 1, f );

        som = brly_som::Som<float>(rows, cols, vector_dim);
        uint t;
        rc += fread(&t, sizeof(uint), 1, f);
        som.setT(t);

        for(int r = 0 ; r < rows ; r++){
            for(int c = 0 ; c < cols ; c++){

                float fl[vector_dim];

                rc += fread(fl, sizeof(float), vector_dim, f);

                som.data()[r][c].clear();
                for(int i = 0; i < vector_dim ; i++){
                    som.data()[r][c].push_back(fl[i]);
                }
            }
        }

        fclose(f);

        size_t should_read = 4 + rows * cols  * vector_dim;

        if(rc != should_read){
            fprintf(stderr, "Trouble loading file %s\n", filename.c_str());
            fprintf(stderr, "Expected %lu, but loaded %lu\n", should_read, rc);
            return false;
        }
        return true;

    }

};

#endif
