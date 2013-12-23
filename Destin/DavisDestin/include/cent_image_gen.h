#ifndef CENT_IMAGE_GEN_H
#define CENT_IMAGE_GEN_H

/* Functions for generating centroid images.
 * Works in a recursive fashion. Layer 0 centroids directly
 * represent their images. For upper layers, each centroid image
 * represents a mixing of child centroid images.
 */

float **** Cig_CreateCentroidImages(Destin * d, float weighParameter);

void Cig_DestroyCentroidImages(Destin * d, float **** images);

void Cig_UpdateCentroidImages(Destin * d, float **** images, float weightParameter);

int Cig_GetCentroidImageWidth(Destin * d, int layer);

void Cig_PowerNormalize(float * weights_in, float * weights_out, int len, float exponent);

#endif
