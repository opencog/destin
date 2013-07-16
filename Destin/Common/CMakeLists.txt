include_directories(
    ../DavisDestin/include
    ../TreeMining
    ../TreeMining/CMOrderedTreeMiner
)

add_library( common SHARED
    DestinIterationFinishedCallback.h
    DestinNetworkAlt
    INetwork.h
    Transporter.h
    VideoSource
    unit_test.h
    CifarSource.h
    BeliefExporter.h
    ImageSourceBase.h
    ImageSourceImpl.h
    DestinTreeManager
    czt_lib.h
    czt_lib2.h
)

add_executable( testCifarSource TestCifar.cc )

find_package( OpenCV REQUIRED) 

target_link_libraries(testCifarSource ${OpenCV_LIBS} )

target_link_libraries( common cm_ordered_treeminer_wrapper )
