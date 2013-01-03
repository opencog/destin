
#include <memory.h>
#include <math.h>
#include "macros.h"
#include "destin.h"
#include "cent_image_gen.h"


void BlendImages(float ** images,       // array of images to blend
                 float * weights,       // weights used to blend images ( will be normalized )
                 int nImages,           // number of images to blend
                 const int img_size,    // number of pixels in each image to blend
                 float * image_out){    // blended image is stored here. must be preallocated memory of length img_size

    int p, i;
    float pix_out, sum = 0;
    float norm_weights[nImages];

    memcpy(norm_weights, weights, nImages * sizeof(float));

    for(i = 0 ; i < nImages ; i++){
        sum += norm_weights[i];
    }

    for(i = 0 ; i < nImages ; i++){
        norm_weights[i] /= sum;
    }

    for(p = 0 ; p < img_size ; p++){
        pix_out = 0;
        for(i = 0 ; i < nImages ; i++){
            pix_out += norm_weights[i] * images[i][p];
        }
        image_out[p] = pix_out;
    }
    return;
}

void ConcatImages(float ** images,// Array of the 4 images to concat.
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

float*** CreateCentroidImages(Destin * d){

    if(!d->isUniform){
        return NULL;
    }

    // allocate memory for the centroid images
    float *** images;
    MALLOC(images, float **, d->nLayers);
    int l, c, image_width = GetNodeFromDestin(d, 0, 0, 0)->ni;
    for(l = 0 ; l < d->nLayers; l++){
        MALLOC(images[l], float *, d->nb[l]);
        for(c = 0 ; c < d->nb[l]; c++){
            MALLOC(images[l][c], float, image_width * image_width);
        }
        image_width *= 2;
    }

    UpdateCentroidImages(d, images);

    return images;
}

int GetCentroidImageWidth(Destin * d, int layer){

    return sqrt(GetNodeFromDestin(d, 0, 0, 0)->ni) * pow(2, layer);
}

void UpdateCentroidImages(Destin * d, float *** images){

    int p, l, c, prev_image_width,image_width;

    Node * n;
    image_width = sqrt(GetNodeFromDestin(d, 0, 0, 0)->ni);
    prev_image_width = image_width;
    for(l = 0 ; l < 1 ; l++){
        n = GetNodeFromDestin(d, l, 0, 0);

        for(c = 0 ; c < d->nb[l]; c++){
            for(p = 0 ; p < n->ni ; p++){
                images[l][c][p] = n->mu[c * n->ns + p];
            }
        }

        prev_image_width = image_width;
        image_width *= 2;
    }

    int child_section, i;
    for(l = 1 ; l < d->nLayers ; l++){
        n = GetNodeFromDestin(d, l, 0, 0);

        float ** combined_images;
        MALLOC(combined_images, float *, 4);
        for(i = 0 ; i < 4 ; i++){
            MALLOC(combined_images[i], float, prev_image_width * prev_image_width);
        }

        for(c = 0 ; c < d->nb[l]; c++){
            // combine the child layer's shared centroids according
            // to the weights given in
            for(child_section = 0 ; child_section < 4 ; child_section++){

                // For the current sub section of the current centroid,
                // generate its representative image and store it in
                // the appropriate section of combined_images;
                BlendImages(images[l - 1],
                              &n->mu[c * n->ns + child_section * d->nb[l - 1]],
                              d->nb[l - 1],
                              prev_image_width * prev_image_width,
                              combined_images[child_section]);
                }

            ConcatImages(combined_images, prev_image_width, prev_image_width, images[l][c] );
        }

        for(i = 0 ; i < 4 ; i++){
            FREE(combined_images[i]);
            combined_images[i] = NULL;
        }
        FREE(combined_images);
        combined_images = NULL;

        prev_image_width = image_width;
        image_width *= 2;
    }
    return;
}

void DestroyCentroidImages(Destin * d, float *** images){

    int l, c;
    for(l = 0 ; l < d->nLayers ; l++){
        for(c = 0; c < d->nb[l]; c++){
            FREE(images[l][c]);
        }
        FREE(images[l]);
    }
    FREE(images);
    return;
}
