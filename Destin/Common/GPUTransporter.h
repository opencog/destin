#ifndef GPUTransporter_h_
#define GPUTransporter_h_

#include "Transporter.h"
class GPUTransporter : public Transporter {
    private:
        int floatArrayLength;
    protected:
        float * sourceImage;
        float * transformedImage;

    public:
        virtual ~GPUTransporter(){
            cudaFree(dest);
        }

        GPUTransporter(int floatArrayLength )
            :floatArrayLength(floatArrayLength),
            transformedImage(NULL){

                CUDA_TEST_MALLOC( (void**)&dest, floatArrayLength  * sizeof(float) );
            }

        virtual void transport(){
            transform();
            cudaMemcpy( dest, transformedImage, floatArrayLength *sizeof(float), cudaMemcpyHostToDevice );
        }

};
#endif GPUTransporter_h_
