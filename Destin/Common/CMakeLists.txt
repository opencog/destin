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
    VideoWriter
    unit_test.h
    CifarSource.h
    BeliefExporter.h
    ImageSourceBase.h
    ImageSourceImpl.h
    DestinTreeManager

    CztMod.h
)

add_executable( testCifarSource TestCifar.cpp )
add_executable( testDestinNetworkAlt TestDestinNetworkAlt.cpp )

find_package( OpenCV REQUIRED) 

target_link_libraries(testCifarSource ${OpenCV_LIBS})
target_link_libraries(common cm_ordered_treeminer_wrapper destinalt)
target_link_libraries(testDestinNetworkAlt common destinalt)

install(TARGETS common testCifarSource testDestinNetworkAlt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)
