add_subdirectory(CMOrderedTreeMiner)

include_directories(CMOrderedTreeMiner ../Common ../DavisDestin/include)

add_executable(testTreeMiner
TestTreeMiner.cc
CMOrderedTreeMinerWrapper.h
)

target_link_libraries(testTreeMiner common destinalt_test cm_ordered_treeminer)
