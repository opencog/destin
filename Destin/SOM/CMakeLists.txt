PROJECT( SOM )

include_directories( brly/som )


#add_subdirectory(som_pak-3.1)
add_subdirectory( brly/som )

add_library(som_common SHARED
    ISom
    BrlySom
    brly/som/brly_som.hpp
    SomPresentor
    cluster/src/cluster
    ClusterSom.hpp
)

target_link_libraries(som_common ml cvaux highgui cv cxcore)

SET_TARGET_PROPERTIES(som_common PROPERTIES LINKER_LANGUAGE CXX)

add_executable( testSOM TestSom.cc)
target_link_libraries(testSOM som_common)
