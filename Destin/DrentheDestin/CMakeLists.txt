#The name of the project
PROJECT(DretheDestin)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# uncomment to be in Release mode
# SET(CMAKE_BUILD_TYPE Release)

# uncomment to build in debug mode
# SET(CMAKE_BUILD_TYPE Debug)

IF (CMAKE_BUILD_TYPE STREQUAL "")
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

IF(CMAKE_BUILD_TYPE STREQUAL "Debug" )
    SET(dbg_flags -g -G)
ENDIF()

INCLUDE_DIRECTORIES( ../Common )

#Searching CUDA
FIND_PACKAGE(CUDA 4.0)

if(CUDA_FOUND)
    #Enable debug for CUDA and set Tesla build command
    #Change the 20 for the type of you card: (CUDA computable of 1.3 = 13)
    set(CUDA_NVCC_FLAGS ${dbg_flags} -gencode arch=compute_30,code=sm_30)
    
    #Add Pugi XML
    add_subdirectory( pugixml )
            
    #Add cuRAND libary of CUDA
    if(WIN32)
            #find_library_local_first is nifty macro found in cmake's FindCUDA script 
            find_library_local_first(lib_curand curand "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/lib/win32")    
    else()
               FIND_LIBRARY(lib_curand curand "/usr/local/cuda/lib64" )
    endif()

    set(cudadest_source DestinCuda.cu DestinData.cu AdviceData.cu DestinKernel.cu )

    CUDA_ADD_LIBRARY(cudadestin SHARED ${cudadest_source} ) 
    CUDA_ADD_EXECUTABLE(destinCuda DestinCuda.cu)

    TARGET_LINK_LIBRARIES( destinCuda
        pugixml
        ${lib_curand}
        cudadestin
        ${lib_highgui}
    )

    if(WIN32)
        link_directories( $ENV{CUDA_LIB_PATH} )
    endif()
    
else()
    MESSAGE("CUDA not found. Not building CUDA DestinPort.")
endif()
