include_directories(
    ../DavisDestin/include
    ../TreeMining
    ../TreeMining/CMOrderedTreeMiner
)

add_library( common SHARED
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
    DestinTreeManager

    # 2013.4.19
    # CZT
    #
    #VideoSource_c1.h
    #VideoSource_c1.cpp
)

add_executable( testCifarSource TestCifar.cc )

find_package( OpenCV REQUIRED) 

target_link_libraries(testCifarSource ${OpenCV_LIBS} )

target_link_libraries( common cm_ordered_treeminer_wrapper )
