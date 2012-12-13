#ifndef SOM_HPP
#define SOM_HPP

#include "ISom.hpp"

#include "brly_som.hpp"

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>

class Som : public ISom {

    brly_som::Som<float> som;

    cv::Mat sim_map;            //simularity map
    cv::Mat resized_sim_map;    //scaled visually
    cv::Mat draw_on_sim_map;    //where user shapes are drawn

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

    typedef struct {
        uint row;
        uint col;
        float hue; //0 to 1
        uint size; //radius or width

    } map_marker;

    std::vector<map_marker> map_markers;

    // Draws the given map marker to the given cv::Mat.
    // The map marker coordinates are relative to this SOM's size
    // and rescaled to the given mat's size so the location,
    // if expressed as a percentage of width and height, is constant.
    void drawMarker(cv::Mat & mat, map_marker & mm){
            cv::Point p;
            p.y = mm.row * (double)mat.rows / (double)rows;
            p.x = mm.col * (double)mat.cols / (double)cols;
            float r, g, b;
            float hue = mm.hue * 360.0, sat = 1.0, val = 1.0;
            HSVtoRGB(&r, &g, &b, hue, sat, val );
            printf("R: %f, G: %f, B: %f\n", r, g, b);
            cv::circle(mat, p, mm.size ,cv::Scalar(r*255, g*255, b*255), -1);
    }



public:



    Som(int map_rows, int map_cols, int vector_dim)
        :rows(map_rows),
          cols(map_cols),
          vector_dim(vector_dim),
          som(map_rows, map_cols, vector_dim),
          sim_map(map_rows, map_cols, CV_32FC1)
    { }

    void train_iterate(float * data){
        som.next(data);
    }

    //taken from http://www.cs.rit.edu/~ncs/color/t_convert.html#RGB to HSV & HSV to RGB
    // r,g,b values are from 0 to 1
    // h = [0,360], s = [0,1], v = [0,1]
    //		if s == 0, then h = -1 (undefined)
    void HSVtoRGB( float *r, float *g, float *b, float h, float s, float v )
    {
        //h is 0 to 360
        //s is 0 to 1
        //v is 0 to 1
            int i;
            float f, p, q, t;
            if( s == 0 ) {
                    // achromatic (grey)
                    *r = *g = *b = v;
                    return;
            }
            h /= 60;			// sector 0 to 5
            i = floor( h );
            f = h - i;			// factorial part of h
            p = v * ( 1 - s );
            q = v * ( 1 - s * f );
            t = v * ( 1 - s * ( 1 - f ) );
            switch( i ) {
                    case 0:
                            *r = v;
                            *g = t;
                            *b = p;
                            break;
                    case 1:
                            *r = q;
                            *g = v;
                            *b = p;
                            break;
                    case 2:
                            *r = p;
                            *g = v;
                            *b = t;
                            break;
                    case 3:
                            *r = p;
                            *g = q;
                            *b = v;
                            break;
                    case 4:
                            *r = t;
                            *g = p;
                            *b = v;
                            break;
                    default:		// case 5:
                            *r = v;
                            *g = p;
                            *b = q;
                            break;
            }
    }

    cv::Mat convertFtoU(cv::Mat & src, cv::Mat & dst){

        cv::Mat ret(src.size(),CV_8UC3 );

        float * data = (float *)src.data;
        for(int i = 0 ; i < src.rows * src.cols; i++){
            //ret.data
        }
        src.convertTo(dst, CV_8UC3);
    }

    void showSimularityMap(string windowName = "SOM Simularity Map  ", uint nh_width = 2, int window_width = 512, int window_height = 512){
        calcSimularityMap(nh_width);

        cv::Size size(window_width, window_height);

        cv::resize(sim_map, resized_sim_map, size);

        cv::Mat todraw;
        if(map_markers.size() == 0){
            todraw = resized_sim_map;
        }else{
            cv::Mat recolored(size, CV_8UC3);
            //cvtColor()
            resized_sim_map.convertTo(recolored, CV_8UC3, 255.0);
            cv::Mat temp;
            cvtColor(recolored, temp, CV_GRAY2RGB);

            //resized_sim_map.convertTo(recolored, CV_8UC3);
            for(int m  = 0 ; m < map_markers.size() ; m++){
                drawMarker(temp, map_markers[m]);
            }
            todraw = temp;
        }



        cv::imshow(windowName, todraw);
        cv::waitKey(5);
    }

    void clearSimMapMarkers(){
        map_markers.clear();
    }

    void addSimMapMaker(uint row, uint col, float hue, uint marker_width){
        map_marker mm;
        mm.col = col;
        mm.row = row;
        mm.size = marker_width;
        mm.hue = hue;
        map_markers.push_back(mm);
    }

    CvPoint findBestMatchingUnit(float * data){
        int r, c;
        som.find_bmu(data, r, c);
        CvPoint p;
        p.x = c;
        p.y = r;
        return p;
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
