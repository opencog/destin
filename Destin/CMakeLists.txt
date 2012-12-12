cmake_minimum_required(VERSION 2.8)

PROJECT(Destin)
# uncomment to be in Release mode
SET(CMAKE_BUILD_TYPE Release)
 
# uncomment to build in debug mode
#SET(CMAKE_BUILD_TYPE Debug)
    
ADD_SUBDIRECTORY(Common)
ADD_SUBDIRECTORY(DavisDestin)
ADD_SUBDIRECTORY(DrentheDestin)
ADD_SUBDIRECTORY(Bindings)
ADD_SUBDIRECTORY(Misc)
ADD_SUBDIRECTORY(SOM)
