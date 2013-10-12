#The name of the project
PROJECT(DavisDestin)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# uncomment to be in Release mode
# SET(CMAKE_BUILD_TYPE Release)

# uncomment to build in debug mode
# SET(CMAKE_BUILD_TYPE Debug)

IF (CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Release)
ENDIF()


FIND_PACKAGE( OpenMP REQUIRED )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


FIND_PACKAGE( OpenCV REQUIRED)

INCLUDE_DIRECTORIES( include ../Common )


ADD_EXECUTABLE(destin src/destin_mnist.c )    

set(destalt_sources
    src/destin.c
    include/node.h
    src/node.c
    src/util.c
    src/cent_image_gen.c
    include/macros.h
    include/node.h
    include/cent_image_gen.h
    include/destin.h
    src/learn_strats.c
    include/learn_strats.h
    include/belief_transform.h
    src/belief_transform.c
)

ADD_LIBRARY( destinalt SHARED ${destalt_sources})
TARGET_LINK_LIBRARIES( destinalt ${OpenCV_LIBS})

ADD_LIBRARY( destinalt_test SHARED ${destalt_sources})
TARGET_LINK_LIBRARIES( destinalt_test ${OpenCV_LIBS} m)
SET_TARGET_PROPERTIES(destinalt_test PROPERTIES COMPILE_DEFINITIONS "UNIT_TEST")


TARGET_LINK_LIBRARIES( destin destinalt )
SET_TARGET_PROPERTIES( destin PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin )


ADD_EXECUTABLE( test test/test.c )
TARGET_LINK_LIBRARIES( test destinalt_test )
SET_TARGET_PROPERTIES( test PROPERTIES RUNTIME_OUTPUT_DIRECTORY test )

install( TARGETS test destin destinalt_test destinalt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static )
