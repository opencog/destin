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
    CztMod.h
)

add_executable( testCifarSource TestCifar.cpp )
add_executable( testDestinNetworkAlt TestDestinNetworkAlt.cpp )

find_package( OpenCV REQUIRED) 

target_link_libraries(testCifarSource ${OpenCV_LIBS})
target_link_libraries(common cm_ordered_treeminer_wrapper)
target_link_libraries(testDestinNetworkAlt common destinalt)
