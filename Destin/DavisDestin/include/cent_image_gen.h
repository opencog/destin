

#ifndef CENT_IMAGE_GEN_H
#define CENT_IMAGE_GEN_H

float *** Cig_CreateCentroidImages(Destin * d);

void Cig_DestroyCentroidImages(Destin * d, float *** images);

void Cig_UpdateCentroidImages(Destin * d, float *** images);

int Cig_GetCentroidImageWidth(Destin * d, int layer);

void Cig_SetPowerNormalizePower(float power);



#endif
