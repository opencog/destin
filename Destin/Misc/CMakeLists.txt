INCLUDE_DIRECTORIES( ../Common )

ADD_EXECUTABLE( testVideo TestVideo ../Common/VideoSource )
TARGET_LINK_LIBRARiES( testVideo highgui cv cxcore ) 


INCLUDE_DIRECTORIES( ../DavisDestin/include )
ADD_EXECUTABLE( webcamExp WebcamExperiment ../Common/VideoSource )
TARGET_LINK_LIBRARiES( webcamExp highgui cv cxcore destinalt avutil) 

