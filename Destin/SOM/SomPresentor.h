#ifndef SOM_PRESENTOR_HPP
#define SOM_PRESENTOR_HPP


#include <stdio.h>
#include <vector>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ISom.h"

/** This class displays the underlying SOM as an image to the user
  * Can show a simularity map and draw markers using opencv highgui
  */
class SomPresentor {

    ISom & som;

    cv::Mat sim_map;            //simularity map
    cv::Mat resized_sim_map;    //scaled visually
    cv::Mat draw_on_sim_map;    //where user shapes are drawn

    uint rows;
    uint cols;
    uint vector_dim;

    bool inverted;

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



        double distTotal;
        double maxDistance = -1.0;

        // For each point on the SOM, calcuate its simularity to all its
        // neighbors and use the total simularity value as a grayscale pixel
        // for the grayscale simularity map.
        float * sim_data = (float *)sim_map.data;//simularity map raw data
        for(r = 0; r < rows ; r++){
            for(c = 0; c < cols; c++){
                distTotal = 0;
                for(nr = r - nh_width ; nr < r + nh_width ; nr++){
                    if(nr >= 0 && nr  < rows){
                        for(nc = c - nh_width ; nc < c + nh_width ; nc++){
                            if(nc >= 0 && nc < cols){
                                distTotal += som.distance_coords(nr, nc, r, c);
                            }
                        }
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
        if(inverted){
            for(int i = 0 ; i < pixels ; i++){
                sim_data[i] = 1.0 - sim_data[i];
            }
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
    /** Converts a HSV color to RGB
    * r,g,b values are from 0 to 1
    * h = [0,360], s = [0,1], v = [0,1]
                if s == 0, then h = -1 (undefined)
    * taken from http://www.cs.rit.edu/~ncs/color/t_convert.html
    */
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


public:

    SomPresentor(ISom & som)
        :som(som),
          rows(som.cell_rows()),
          cols(som.cell_cols()),
          vector_dim(som.cell_dim()),
          inverted(false)

    {
        sim_map = cv::Mat(rows, cols, CV_32FC1);
        printf("rows: %i, cols: %i, som_rows: %i, som_cols: %i\n", rows, cols, som.cell_rows(), som.cell_cols());

    }


    /** Inverts the simularity map grayscale image.
      * Defaults to false.
      */
    void setInverted(bool inverted){
        this->inverted = inverted;
    }


    void clearSimMapMarkers(){
        map_markers.clear();
    }

    /** Add a marker to be drawn on the simularity map
      * @param hue -  The hue in HSV color space. between 0 and 1 ( mapped to 0 to 360 degrees).
      * @param row - y coordinate on the map to be drawn. In SOM coordinates not SOM display coordinates,
      *                so it rescales itself based on the size of the output window
      * @param col - x coordinate
      * @param marker_width - the radius of the marker, in SOM display coordinates
      */
    void addSimMapMaker(uint row, uint col, float hue, uint marker_width){
        map_marker mm;
        mm.col = col;
        mm.row = row;
        mm.size = marker_width;
        mm.hue = hue;
        map_markers.push_back(mm);
    }

    /** Creates a grayscale representation of the SOM
      * For each cell of the SOM, a square neighborhood of SOM cells around it is chosen.
      * For each cell in the neightboorhood, the distance is taken, then the sum of
      * of distances of the neightborhood is converted into the value of greyscale pixel
      * that represents the cell.
      *
      * Draws markers that have been added with addSimMapMaker on top if it.
      * @param nh_width - 1/2 the width of the square that determintes a cell's neighbors.
      *                   A larger value makes the image smoother, but takes longer to calulate
      *
      * @param window_width and window_height - the simularity map it rescaled to this
      */
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


    void showAndSaveSimularityMap(string fileName = "SOM.jpg", string windowName = "SOM Simularity Map  ",
                                  uint nh_width = 2, int window_width = 512, int window_height = 512){
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
        cv::imwrite(fileName, todraw);
        cv::waitKey(5);
    }

};



#endif
