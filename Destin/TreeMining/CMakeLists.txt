add_subdirectory(CMOrderedTreeMiner)

include_directories(CMOrderedTreeMiner ../Common ../DavisDestin/include .)

add_library(cm_ordered_treeminer_wrapper SHARED
CMOrderedTreeMinerWrapper
)
add_executable(testTreeMiner
TestTreeMiner.cpp
)

if(WIN32)
set(avutil "")
else()
set(avutil avutil)
endif()

target_link_libraries(cm_ordered_treeminer_wrapper cm_ordered_treeminer)
target_link_libraries(testTreeMiner common destinalt_test cm_ordered_treeminer_wrapper ${avutil})

install(TARGETS cm_ordered_treeminer_wrapper testTreeMiner
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static )
