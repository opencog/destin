
%module(directors="1") SWIG_MODULE_NAME 
%{
/* includes that are needed to compile */
#include "macros.h"
#include "DestinIterationFinishedCallback.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "VideoSource.h"
#include "Transporter.h"
#include "INetwork.h"
#include "DestinNetworkAlt.h"
#include "GenerativeVisualizer.h"
#include "CifarSource.h"
#include "ISom.hpp"    
#include "BrlySom.hpp"    
#include "SomPresentor.hpp"
#include "BeliefExporter.h"
%}

%include "macros.h"
typedef unsigned int uint;
/* 
turn on director wrapping callback, so c++ code can call methods defined in the target language
See http://www.swig.org/Doc2.0/SWIGDocumentation.html#Java_directors
See https://swig.svn.sourceforge.net/svnroot/swig/trunk/Examples/java/callback/
*/

%feature("director") DestinIterationFinishedCallback;
%include "DestinIterationFinishedCallback.h"


/* lets you use java strings easily with c++ strings */
%include "std_string.i"

/* be able to use INetwork as an abstract interface in Java */
%feature("director") INetwork; 
%include "INetwork.h"

/* the other classes to generate wrappers for */
%include "destin.h"
%include "node.h"
%include "VideoSource.h"
%include "Transporter.h"
%include "DestinNetworkAlt.h"
%include "GenerativeVisualizer.h"
%include "learn_strats.h"
%include "CifarSource.h"
%include "ISom.hpp"
%include "BrlySom.hpp"
%include "SomPresentor.hpp"
%include "BeliefExporter.h"

/* carrays.i so you can use a c++ pointer like an array */
%include "carrays.i" 
%array_class(int, SWIG_IntArray);
%array_class(float, SWIG_FloatArray);
%array_functions(float *, SWIG_Float_p_Array);
%array_class(uint, SWIG_UInt_Array);
%array_functions(Node *, SWIG_Node_p_Array);
%array_class(Node, SWIG_NodeArray);

/* some opencv functions */
namespace cv {
class Mat;
void imshow( const string& winname, const Mat& mat );
}
typedef struct CvPoint
{
      int x;
      int y;
}
CvPoint;
