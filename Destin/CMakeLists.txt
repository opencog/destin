cmake_minimum_required(VERSION 2.8)

PROJECT(Destin)
# uncomment to be in Release mode
SET(CMAKE_BUILD_TYPE Release)
 
# uncomment to build in debug mode
#SET(CMAKE_BUILD_TYPE Debug)

#SET(CMAKE_EXE_LINKER_FLAGS -pg)
#ADD_DEFINITIONS( -pg)

####

if(WIN32)
set(SWIG_EXECUTABLE    C:/swigwin-2.0.11/swig.exe)
set(OpenCV_DIR  C:/opencv-2.4.6/mingwbuild/install)
set(CMAKE_MODULE_PATH  C:/opencv-2.4.6/mingwbuild/install)
set(CMAKE_LIBRARY_PATH C:/opencv-2.4.6/mingwbuild/install/lib)
INSTALL(FILES mingw_run_test.bat DESTINATION bin)
endif()
####

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install)

FIND_PACKAGE( OpenCV REQUIRED)
ADD_SUBDIRECTORY(TreeMining)
ADD_SUBDIRECTORY(Common)
ADD_SUBDIRECTORY(DavisDestin)
#ADD_SUBDIRECTORY(DrentheDestin) # Requires CUDA to build.
ADD_SUBDIRECTORY(Bindings)
ADD_SUBDIRECTORY(Misc)
ADD_SUBDIRECTORY(SOM)

# 2013.4.11
# CZT
ADD_SUBDIRECTORY(czt_test)
