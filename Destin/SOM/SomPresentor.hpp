#ifndef SOM_PRESENTOR_HPP
#define SOM_PRESENTOR_HPP

#include <vector>
#include <math.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include "cluster/src/cluster.h"
#include "ISom.hpp"



//prototype from SOM/cluster/src/cluster.h
//can't include it because that defines Node
/*
void somcluster (int nrows, int ncolumns, double** data, int** mask,
      const double weight[], int transpose, int nxnodes, int nynodes,
      double inittau, int niter, char dist, double*** celldata,
      int clusterid[][2]);
*/
class SomPresentor {


    ISom & som;

    cv::Mat sim_map;            //simularity map
    cv::Mat resized_sim_map;    //scaled visually
    cv::Mat draw_on_sim_map;    //where user shapes are drawn

    uint rows;
    uint cols;
    uint vector_dim;

    typedef struct {
        uint row;
        uint col;
        float hue; //0 to 1
        uint size; //radius or width

    } map_marker;

    std::vector<map_marker> map_markers;


    void calcSimularityMap(uint nh_width = 2){

        unsigned int r,//row
                     c;//col
        int nr, //neigbor row
            nc; //neighbor col (can be negative if near the border)


        float * neighbor;
        float * origin;

        double distTotal;
        double maxDistance = -1.0;

        // For each point on the SOM, calcuate its simularity to all its
        // neighbors and use the total simularity value as a grayscale pixel
        // for the grayscale simularity map.
        float * sim_data = (float *)sim_map.data;//simularity map raw data
        for(r = nh_width ; r < rows - nh_width; r++){
            for(c = nh_width; c < cols - nh_width; c++){
                origin =  som.getMapCell(r, c);
                distTotal = 0;
                for(nr = r - nh_width ; nr < r + nh_width ; nr++){
                    if(nr < 0 || nr >= rows){ continue; }

                    for(nc = c - nh_width ; nc < c + nh_width ; nc++){
                        if(nc < 0 || nc >= cols){ continue; }
                        neighbor = som.getMapCell(nr, nc);
                        distTotal += som.distance(origin, neighbor);
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
            cv::circle(mat, p, mm.size ,cv::Scalar(r*255, g*255, b*255), -1);
    }


public:

    SomPresentor(ISom & som)
        :som(som),
          rows(som.cell_rows()),
          cols(som.cell_cols()),
          vector_dim(som.cell_dim())

    {
        sim_map = cv::Mat(rows, cols, CV_32FC1);
        printf("rrows: %i, cols: %i, som_rows: %i, som_cols: %i\n", rows, cols, som.cell_rows(), som.cell_cols());

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

    void showSimularityMap(string windowName = "SOM Simularity Map  ", uint nh_width = 2, int window_width = 512, int window_height = 512){
        calcSimularityMap(nh_width);

        cv::Size size(window_width, window_height);

        cv::resize(sim_map, resized_sim_map, size);

        cv::Mat todraw;
        if(map_markers.size() == 0){
            todraw = resized_sim_map;
        }else{
            cv::Mat recolored(size, CV_8UC3);
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


};



#endif
