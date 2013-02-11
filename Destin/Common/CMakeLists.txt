include_directories( ../DavisDestin/include )
ADD_LIBRARY( common SHARED
    DestinIterationFinishedCallback.h
    DestinNetworkAlt
    GenerativeVisualizer
    INetwork.h
    Transporter.h
    VideoSource
    unit_test.h
    CifarSource.h
    BeliefExporter.h
    ImageSourceBase.h
    ImageSourceImpl.h
)


add_executable( testCifarSource TestCifar.cc )

find_package( OpenCV REQUIRED) 
target_link_libraries(testCifarSource ${OpenCV_LIBS} )
