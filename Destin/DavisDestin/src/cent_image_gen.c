
#include <memory.h>
#include <math.h>
#include "macros.h"
#include "destin.h"
#include "cent_image_gen.h"

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

void Cig_BlendImages(float ** images,   // array of images to blend
                 float * weights,       // weights used to blend images ( will be normalized )
                 int nImages,           // number of images to blend
                 const int img_size,    // number of pixels in each image to blend
                 float weighParameter,  // a value higher than one makes for more contrasting images
                 float * image_out,     // blended image is stored here. must be preallocated memory of length img_size
                 int layer,
                 int extRatio){

    int p, i;
    float pix_out;

    float * w;
    float norm_weights[nImages];
    Cig_PowerNormalize(weights, norm_weights, nImages, weighParameter);
    w = norm_weights;

    if(layer != 1)
    {
        for(p = 0 ; p < img_size ; p++){
            pix_out = 0;
            for(i = 0 ; i < nImages ; i++){
                pix_out += w[i] * images[i][p];
            }
            image_out[p] = pix_out;
        }
    }
    else
    {
        for(p = 0 ; p < img_size ; p++){
            pix_out = 0;
            for(i = 0 ; i < nImages ; i++){
                float tempP = images[i][p];
                int j;
                for(j=1; j<extRatio; ++j)
                {
                    tempP += images[i][p + img_size*j];
                }
                pix_out += w[i]*tempP;
            }
            image_out[p] = pix_out;
        }
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

float*** Cig_CreateCentroidImages(Destin * d, float weightParameter){

    if(!d->isUniform){
        return NULL;
    }

    // allocate memory for the centroid images
    float *** images;
    MALLOC(images, float **, d->nLayers);
    int l, c, image_width = sqrt(GetNodeFromDestin(d, 0, 0, 0)->ni);
    for(l = 0 ; l < d->nLayers; l++){
        MALLOC(images[l], float *, d->nb[l]);
        for(c = 0 ; c < d->nb[l]; c++){
            MALLOC(images[l][c], float, (l==0 ? image_width*image_width*d->extRatio : image_width*image_width));
        }
        image_width *= 2;
    }

    Cig_UpdateCentroidImages(d, images, weightParameter);

    return images;
}

int Cig_GetCentroidImageWidth(Destin * d, int layer){

    return sqrt(GetNodeFromDestin(d, 0, 0, 0)->ni) * pow(2, layer);
}


static void _Cig_UpdateLayerZeroImages(Destin * d,
                                       float *** images    // preallocated images to update
                                       ){

    int p, c;
    Node * n = GetNodeFromDestin(d, 0, 0, 0);

    for(c = 0 ; c < d->nb[0]; c++){
        for(p = 0 ; p < n->ni ; p++){
            images[0][c][p] = n->mu[c][p];
        }

        int ext_ratio;
        for(ext_ratio = 1; ext_ratio < d->extRatio; ++ext_ratio){
            for(p=0; p < n->ni; ++p){
                images[0][c][n->ni*ext_ratio + p] = n->mu[c][n->ni*ext_ratio + n->nb + n->np + p];
            }
        }
    }
}

void Cig_UpdateCentroidImages(Destin * d,
                              float *** images,    // preallocated images to update
                              float weighParameter // higher value means higher contrast
                              ){

    int child_section, i, l, c;

    _Cig_UpdateLayerZeroImages(d, images);

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
                Cig_BlendImages(images[l - 1],
                              &n->mu[c][child_section * d->nb[l - 1]],
                              d->nb[l - 1],
                              child_image_width * child_image_width,
                              weighParameter,
                              combined_images[child_section],
                              l,
                              d->extRatio);
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

void Cig_DestroyCentroidImages(Destin * d, float *** images){

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
