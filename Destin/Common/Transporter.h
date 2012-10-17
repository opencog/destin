#ifndef TRANSPORTER_H_
#define TRANSPORTER_H_

#include <iostream>
#include <stdexcept>
/** Transforms a source image and then transfers it to its destination.
 *
 * This base class does not do much, just sets the destination as the source and
 * does no transformations.
 *
 */

#define CUDA_TEST_MALLOC( p, s )                                                                    \
	if( cudaMalloc( p , s ) != 0 ){                                                                 \
		 stringstream mess; mess << "could not cudaMaclloc at " << __FILE__ << ":" << __LINE__ ;    \
	     throw runtime_error(mess.str());}


using namespace std;
class Transporter {
protected:
	float * destImage;
	float * sourceImage;
	float * transformedImage;

	/**
	 * Inheriting classes should overide this to
	 * transform the image as necessary setting the
	 * transformedImaged.
	 */
	virtual void transform(){
		transformedImage = sourceImage;
	}

public:
	virtual ~Transporter(){}

	Transporter(){}

	void setSource(float * sourceImage){
		this->sourceImage = sourceImage;
	}

	float * getDest(){
		return destImage;
	}

	/**x
	 * inheriting classes should override this to
	 * call transform and set the destinationImage
	 */
	virtual void transport(){
		transform();
		destImage = transformedImage;
	}

};


/** Transforms the image to align with node regions.
 * Drente destin needs this, DavisDestin does not because it already
 * arranges the input for its nodes.
 */
class ImageTransporter : public Transporter {
private:
	const int nodes_wide;
	const int nodes_high;
	const int ppn_x; //pixels per node in the x direction
	const int ppn_y; //pixels per node in the y direction

protected:
	/**
	 * transform - takes the source image, then rearranges it so that
	 * the DeSTIN input nodes will each have a ppnx by ppny image input region
	 */
	virtual void transform(){
		const int image_width = nodes_wide * ppn_x;
		int i = 0;
		int imx; //image x
		int imy; //image y

		for(int ny = 0 ; ny < nodes_high ; ny++ ){
			for(int nx = 0 ; nx < nodes_wide ; nx++ ){
				for(int y  = 0 ; y < ppn_y ; y++ ){
					for(int x = 0 ; x < ppn_x ; x++ ){
						imx = nx * ppn_x + x;
						imy = ny * ppn_y + y;
						transformedImage[i] = sourceImage[image_width * imy + imx];
						i++;
					}
				}
			}
		}//end for ny
	}// end transform
public:

	virtual ~ImageTransporter(){
		delete [] transformedImage;
	}
	ImageTransporter(int nodes_wide, int nodes_high, int ppn_x, int ppn_y)
		:nodes_wide(nodes_wide),
		 nodes_high(nodes_high),
		 ppn_x(ppn_x),ppn_y(ppn_y) {
	    transformedImage = new float[nodes_wide * nodes_high * ppn_x * ppn_y];
	}

};
#endif //TRANSPORTER_H_
