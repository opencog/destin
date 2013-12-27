
#include <memory.h>
#include <math.h>
#include "macros.h"
#include "destin.h"
#include "cent_image_gen.h"

static void _Cig_UpdateCentroidImages(Destin * d,
                                      float *** images,      // preallocated images to update
                                      float weightParameter, // higher value means higher contrast
                                      int channel);

static void _Cig_UpdateLayerZeroImages(Destin * d,
                                       float *** images, int channel);   // preallocated images to update

void Cig_Normalize(float * weights_in, float * weights_out, int len, float not_used){
    memcpy(weights_out, weights_in, len * sizeof(float));
    int i;
    float sum;
    for(i = 0 ; i < len ; i++){
        sum += weights_out[i];
    }
    for(i = 0 ; i < len ; i++){
        weights_out[i] /= sum;
    }
}

void Cig_PowerNormalize(float * weights_in, float * weights_out, int len, float exponent){
    int i;
    if(weights_out != weights_in){
        memcpy(weights_out, weights_in, len * sizeof(float));
    }

    float sum = 0;
    if(exponent == 1.0){
        for(i = 0 ; i < len ; i++){
            sum+=weights_out[i];
        }
    }else{
        for(i = 0 ; i < len ; i++){
            weights_out[i] = pow(weights_out[i],exponent);
            sum+=weights_out[i];
        }
    }
    for(i = 0 ; i < len ; i++){
        weights_out[i]/=sum;
    }
}

static void _Cig_BlendImages(float ** images,   // array of images to blend
                 float * weights,       // weights used to blend images ( will be normalized )
                 int nImages,           // number of images to blend
                 const int img_size,    // number of pixels in each image to blend
                 float weighParameter,  // a value higher than one makes for more contrasting images
                 float * image_out      // blended image is stored here. must be preallocated memory of length img_size
                 ){

    int pixel, i;
    float pix_out;

    float norm_weights[nImages];
    Cig_PowerNormalize(weights, norm_weights, nImages, weighParameter);

    for(pixel = 0 ; pixel < img_size ; pixel++){
        pix_out = 0;
        for(i = 0 ; i < nImages ; i++){
            pix_out += norm_weights[i] * images[i][pixel];
        }
        image_out[pixel] = pix_out;
    }

    return;
}

//TODO: update to handle nodes that have more than 4 children
void Cig_ConcatImages(float ** images,// Array of the 4 images to concat.
                  const int rows, // Height of each image to concat.
                  const int cols, // Width of each image to concat.
                  float * image_out // Concated image that is 2 x wider and taller
                                    // than the inividual images.
                  ){

    int r, p = 0;
    for(r = 0 ; r < rows ; r++){
        memcpy(&image_out[p], &images[0][r * cols], sizeof(float) * cols);
        p+=cols;
        memcpy(&image_out[p], &images[1][r * cols], sizeof(float) * cols);
        p+=cols;
    }
    for(r = 0 ; r < rows ; r++){
        memcpy(&image_out[p], &images[2][r * cols], sizeof(float) * cols);
        p+=cols;
        memcpy(&image_out[p], &images[3][r * cols], sizeof(float) * cols);
        p+=cols;
    }
}

/**
 * Allocates memory to hold the centroid images.
 * Then calls Cig_UpdateCentroidImages to generate them.
 *
 * The returned images is a 4 dimenional array index by
 * images[channel][layer][centroid][pixel].
 *
 * Range of the indicies:
 * channel: 0 to d->extRatio - 1
 * layer: 0 to destin->nLayers - 1
 * centroid: 0 to destin->nb[layer] - 1
 * pixel: 0 to number of pixels in that centroid image
 *
 * The channel can represent color, or a side in a steroscopic image.
 * For example, if dealing with only 1 grayscale image, then there's only 1 channel,
 * and extRatio = 1. If dealing with color images, then there may be 3 channels,
 * extRatio = 3, one channel for each Red, Green, Blue. If dealing with a pair of
 * color images for color 3D vision, then there may be 6 channels, extRatio = 6,
 * representing a pair of RGB images.
 *
 * This code generates centroid images for each channel independently.
 * Then it is up to calling code to combine the channels into one image.
 * For example, combine 3 R,G,B channels to show color images to the user.
 *
 * User should deallocate the images with Cig_DestroyCentroidImages function.
 *
 * @brief Cig_CreateCentroidImages
 * @param d
 * @param weightParameter
 * @return generated images.
 */
float**** Cig_CreateCentroidImages(Destin * d, float weightParameter){

    if(!d->isUniform){
        return NULL;
    }

    // allocate memory for the centroid images
    float**** images;
    MALLOC(images, float ***, d->extRatio); // allocate for each channel

    int channel, l, c, image_width;
    for(channel = 0 ; channel < d->extRatio ; channel++){
        MALLOC(images[channel], float **, d->nLayers);
        image_width = sqrt(GetNodeFromDestin(d, 0, 0, 0)->ni);
        for(l = 0 ; l < d->nLayers; l++){
            MALLOC(images[channel][l], float *, d->nb[l]);
            for(c = 0 ; c < d->nb[l]; c++){
                MALLOC(images[channel][l][c], float, image_width*image_width);
            }
            image_width *= 2;
        }
        _Cig_UpdateCentroidImages(d, images[channel], weightParameter, channel);
    }
    return images;
}

int Cig_GetCentroidImageWidth(Destin * d, int layer){

    return sqrt(GetNodeFromDestin(d, 0, 0, 0)->ni) * pow(2, layer);
}


static void _Cig_UpdateLayerZeroImages(Destin * d,
                                       float *** images,    // preallocated images to update
                                       int channel
                                       ){
    int pixel, c;
    Node * n = GetNodeFromDestin(d, 0, 0, 0);
    for(c = 0 ; c < d->nb[0]; c++){
        for(pixel = 0 ; pixel < n->ni ; pixel++){
            //images[layer][centroid][pixel]
            images[0][c][pixel] = n->mu[c][pixel]; // layer 0 centroids directly represent their images
        }

        if(channel > 0){
            for(pixel=0; pixel < n->ni; ++pixel){
                images[0][c][pixel] = n->mu[c][n->ni*channel + n->nb + n->np + pixel];
            }
        }
    }
}

/** Updates centroid images for all channels
 * @param d
 * @param images
 * @param weightParameter
 */
void Cig_UpdateCentroidImages(Destin *d,
                              float ****images,
                              float weightParameter){
    int channel;
    for(channel = 0 ; channel < d->extRatio ; channel++){
        _Cig_UpdateCentroidImages(d, images[channel], weightParameter, channel);
    }
}

/** Update centroid images for 1 channel
 * @param d
 * @param images
 * @param weighParameter
 */
static void _Cig_UpdateCentroidImages(Destin * d,
                              float *** images,    // preallocated images to update
                              float weightParameter, // higher value means higher contrast
                              int channel
                              ){

    int child_section, i, l, c;

    _Cig_UpdateLayerZeroImages(d, images, channel);

    int child_image_width = sqrt(GetNodeFromDestin(d, 0, 0, 0)->ni);

    for(l = 1 ; l < d->nLayers ; l++){
        Node * n = GetNodeFromDestin(d, l, 0, 0);

        float ** combined_images;
        MALLOC(combined_images, float *, 4);
        for(i = 0 ; i < 4 ; i++){
            MALLOC(combined_images[i], float, child_image_width * child_image_width);
        }

        for(c = 0 ; c < d->nb[l]; c++){
            // combine the child layer's shared centroids according
            // to the weights given in
            for(child_section = 0 ; child_section < 4 ; child_section++){

                // For the current sub section of the current centroid,
                // generate its representative image and store it in
                // the appropriate section of combined_images;
                _Cig_BlendImages(images[l - 1],
                              &n->mu[c][child_section * d->nb[l - 1]],
                              d->nb[l - 1],
                              child_image_width * child_image_width,
                              weightParameter,
                              combined_images[child_section]);
            }

            Cig_ConcatImages(combined_images, child_image_width, child_image_width, images[l][c] );
        }

        for(i = 0 ; i < 4 ; i++){
            FREE(combined_images[i]);
            combined_images[i] = NULL;
        }
        FREE(combined_images);
        combined_images = NULL;

        child_image_width = child_image_width * 2;
    }
    return;
}

void Cig_DestroyCentroidImages(Destin * d, float **** images){

    int l, c, channel;
    for(channel = 0 ; channel < d->extRatio ; channel++){
        for(l = 0 ; l < d->nLayers ; l++){
            for(c = 0; c < d->nb[l]; c++){
                FREE(images[channel][l][c]);
            }
            FREE(images[channel][l]);
        }
        FREE(images[channel]);
    }
    FREE(images);
    return;
}
