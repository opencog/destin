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

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ")


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
    include/array.h
    src/array.c
    include/centroid.h
    src/centroid.c
)

FIND_PACKAGE(OpenCV REQUIRED )

ADD_LIBRARY( destinalt SHARED ${destalt_sources})
TARGET_LINK_LIBRARIES( destinalt ${OpenCV_LIBS} gomp m)

ADD_LIBRARY( destinalt_test SHARED ${destalt_sources})
TARGET_LINK_LIBRARIES( destinalt_test ${OpenCV_LIBS} m gomp)
SET_TARGET_PROPERTIES(destinalt_test PROPERTIES COMPILE_DEFINITIONS "UNIT_TEST")


TARGET_LINK_LIBRARIES( destin destinalt )
SET_TARGET_PROPERTIES( destin PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin )


ADD_EXECUTABLE( tests tests/test.c tests/testconfig.conf)
TARGET_LINK_LIBRARIES( tests destinalt_test )
SET_TARGET_PROPERTIES( tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY tests )

INSTALL( TARGETS tests destin destinalt_test destinalt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static )

INSTALL(FILES test/testconfig.conf DESTINATION bin)

if(WIN32)
    # copy mingw dlls to bin directory
    GET_FILENAME_COMPONENT(MINGW_BIN_DIR ${CMAKE_C_COMPILER} DIRECTORY)
    FILE(GLOB MINGW_DLLS "${MINGW_BIN_DIR}/*.dll")
    INSTALL(FILES ${MINGW_DLLS} DESTINATION bin)
    #copy opencv dlls to bin directory
    INSTALL(DIRECTORY ${OpenCV_DIR}/bin/ DESTINATION bin FILES_MATCHING REGEX "bin/.*dll")
endif(WIN32)
