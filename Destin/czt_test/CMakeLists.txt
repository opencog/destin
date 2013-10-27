###############################################################################
# This is created on 2013.4.5 by CZT.

#CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Name of this project
#PROJECT(czt_test)

#SET(CMAKE_BUILD_TYPE debug)

FIND_PACKAGE(OpenCV REQUIRED)

ADD_LIBRARY( stereovision SHARED 
    include/stereovision.h
    src/stereovision.cpp
    include/stereocamera.h
    src/stereocamera.cpp
)

TARGET_LINK_LIBRARIES(stereovision ${OpenCV_LIBS})

if(WIN32)
set(avutil "") # don't link avutil on windows
else()
set(avutil avutil)
endif()

SET(testlibs common destinalt ${avutil} stereovision som)
    
INCLUDE_DIRECTORIES(../Common ../DavisDestin/include include ../SOM)

#ADD_EXECUTABLE(test1 src/test1.cpp )
#TARGET_LINK_LIBRARIES(test1 ${testlibs})

install(TARGETS stereovision
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static )
