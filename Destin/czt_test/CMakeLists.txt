###############################################################################
# This is created on 2013.4.5 by CZT.

#CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Name of this project
#PROJECT(czt_test)

#SET(CMAKE_BUILD_TYPE debug)

#FIND_PACKAGE(OpenCV REQUIRED)

SET(temp_set
    include/stereovision.h
    src/stereovision.cpp
    include/stereocamera.h
    src/stereocamera.cpp
)

INCLUDE_DIRECTORIES(../Common ../DavisDestin/include include)

ADD_EXECUTABLE(test1 src/test1.cpp  ../Common/VideoSource ${temp_set})
TARGET_LINK_LIBRARIES(test1 ${OpenCV_LIBS} common destinalt avutil)
