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

ADD_LIBRARY( destinalt SHARED
    src/destin.c
    src/node.c
    src/util.c
    src/cent_image_gen.c
    include/macros.h
    include/cent_image_gen.h
    src/learn_strats.c
    include/learn_strats.h
)

TARGET_LINK_LIBRARIES( destinalt ${OpenCV_LIBS})

TARGET_LINK_LIBRARIES( destin destinalt )
SET_TARGET_PROPERTIES( destin PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin )

ADD_EXECUTABLE( test test/test.c )
TARGET_LINK_LIBRARIES( test destinalt )
SET_TARGET_PROPERTIES( test PROPERTIES RUNTIME_OUTPUT_DIRECTORY test )
