INCLUDE_DIRECTORIES( ../Common )

ADD_EXECUTABLE( testVideo TestVideo ../Common/VideoSource )
TARGET_LINK_LIBRARiES( testVideo opencv_highgui opencv_core opencv_imgproc) 


INCLUDE_DIRECTORIES( ../DavisDestin/include )
ADD_EXECUTABLE( webcamExp WebcamExperiment ../Common/VideoSource )
TARGET_LINK_LIBRARiES( webcamExp destinalt opencv_core opencv_imgproc avutil)

