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

INCLUDE_DIRECTORIES( include ../Common )


ADD_EXECUTABLE(destin src/destin_mnist.c )    

ADD_LIBRARY( destinalt SHARED src/destin.c src/node.c src/util.c )
TARGET_LINK_LIBRARIES( destinalt opencv_imgproc opencv_highgui opencv_core)

TARGET_LINK_LIBRARIES( destin destinalt )
SET_TARGET_PROPERTIES( destin PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin )

ADD_EXECUTABLE( test test/test.c )
TARGET_LINK_LIBRARIES( test destinalt )
SET_TARGET_PROPERTIES( test PROPERTIES RUNTIME_OUTPUT_DIRECTORY test )
