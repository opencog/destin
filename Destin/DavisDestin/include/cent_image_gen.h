

#ifndef CENT_IMAGE_GEN_H
#define CENT_IMAGE_GEN_H

float *** CreateCentroidImages(Destin * d);

void DestroyCentroidImages(Destin * d, float *** images);

void UpdateCentroidImages(Destin * d, float *** images);

int GetCentroidImageWidth(Destin * d, int layer);

#endif
