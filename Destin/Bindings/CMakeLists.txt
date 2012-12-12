cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(SWIG 2.0.4)
FIND_LIBRARY(highgui NAMES highgui opencv_highgui)

INCLUDE_DIRECTORIES(
        ../DavisDestin/include
        ../Common 
        ../SOM
        ../SOM/brly/som
)

if(NOT SWIG_FOUND)
    message(FATAL_ERROR "SWIG wasnt found")
endif()
if(NOT highgui)
    message(FATAL_ERROR "highgui not found")
endif()

if(SWIG_FOUND AND highgui)
    INCLUDE(UseSWIG)

    add_subdirectory(Java)
    add_subdirectory(Python)
else()
    message("Could not find either SWIG > 2.0 or libhighgui. Not building language bindings.")
endif()
