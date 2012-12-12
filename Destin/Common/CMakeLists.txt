ADD_LIBRARY( common SHARED
    DestinIterationFinishedCallback.h
    DestinNetworkAlt.h
    GenerativeVisualizer
    INetwork.h
    Transporter.h
    VideoSource
    unit_test.h
    CifarSource.h
    BeliefExporter.h
)


add_executable( testCifarSource TestCifar.cc )

target_link_libraries(testCifarSource highgui cxcore cv)
