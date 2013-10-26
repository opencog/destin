cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(SWIG 2.0.4)

INCLUDE_DIRECTORIES(
        ../DavisDestin/include
        ../Common 
        ../SOM
        ../TreeMining
        ../TreeMining/CMOrderedTreeMiner
)

if(NOT SWIG_FOUND)
    message(FATAL_ERROR "SWIG wasnt found")
endif()

if(SWIG_FOUND)
    INCLUDE(UseSWIG)

    add_subdirectory(Java)
    add_subdirectory(Python)
else()
    message("Could not find either SWIG > 2.0 or libhighgui. Not building language bindings.")
endif()
