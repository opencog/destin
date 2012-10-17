cmake_minimum_required(VERSION 2.8)

#This section creates java JNI bindings using SWIG 2.)0
FIND_PACKAGE(SWIG 2.0.4)
FIND_LIBRARY(highgui highgui)
INCLUDE_DIRECTORIES(../DavisDestin/include ../Common )
if(SWIG_FOUND AND highgui)
    INCLUDE(UseSWIG)

    FIND_PACKAGE(JNI)
    if(JNI_FOUND)
        include_directories( ${JNI_INCLUDE_DIRS} )
        SET_SOURCE_FILES_PROPERTIES(SwigDestinJavaBindings.i PROPERTIES CPLUSPLUS ON)
        SET_SOURCE_FILES_PROPERTIES(SwigDestinJavaBindings.i PROPERTIES SWIG_FLAGS "-package;javadestin"  )
        SET(CMAKE_SWIG_OUTDIR ./gen_src/javadestin )            
        SWIG_ADD_MODULE(destinjava java SwigDestinJavaBindings.i )
        
        SWIG_LINK_LIBRARIES(destinjava highgui destinalt )
    
    else(JNI_FOUND)
        message("Couldn't find jni. Is a jdk installed? Not building java interfaces.")
    endif(JNI_FOUND)
else()
    message("Could not find either SWIG > 2.0 or libhighgui. Not building java interfaces.")
endif()     
