PROJECT( SOM )

add_library(som SHARED
    ISom
    SomPresentor
    cluster/src/cluster
    ClusterSom.hpp
)

target_link_libraries(som ${OpenCV_LIBS})

SET_TARGET_PROPERTIES(som PROPERTIES LINKER_LANGUAGE CXX)

add_executable( testSOM TestSom.cc)
target_link_libraries(testSOM som)
