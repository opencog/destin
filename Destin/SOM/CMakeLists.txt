PROJECT( SOM )

add_library(som SHARED
    ISom
    SomPresentor
    cluster/src/cluster
    ClusterSom.h
)

target_link_libraries(som ${OpenCV_LIBS})

SET_TARGET_PROPERTIES(som PROPERTIES LINKER_LANGUAGE CXX)

add_executable( testSOM TestSom.cpp)
target_link_libraries(testSOM som)

install(TARGETS som testSOM
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static )
