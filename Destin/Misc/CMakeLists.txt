INCLUDE_DIRECTORIES( ../Common )



# don't link avutil_lib with windows
if(WIN32)
set(avutil_lib "")
else()
set(avutil_lib avutil)
endif()

ADD_EXECUTABLE( testVideo TestVideo ../Common/VideoSource )
TARGET_LINK_LIBRARiES( testVideo ${OpenCV_LIBS} ${avutil_lib})


INCLUDE_DIRECTORIES( ../DavisDestin/include )
ADD_EXECUTABLE( webcamExp WebcamExperiment ../Common/VideoSource )
TARGET_LINK_LIBRARiES( webcamExp destinalt ${OpenCV_LIBS} ${avutil_lib} common)

install(TARGETS webcamExp testVideo
    RUNTIME DESTINATION bin)
