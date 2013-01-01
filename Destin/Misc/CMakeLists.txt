INCLUDE_DIRECTORIES( ../Common )

ADD_EXECUTABLE( testVideo TestVideo ../Common/VideoSource )
TARGET_LINK_LIBRARiES( testVideo ${OpenCV_LIBS} avutil)


INCLUDE_DIRECTORIES( ../DavisDestin/include )
ADD_EXECUTABLE( webcamExp WebcamExperiment ../Common/VideoSource )
TARGET_LINK_LIBRARiES( webcamExp destinalt ${OpenCV_LIBS} avutil)

