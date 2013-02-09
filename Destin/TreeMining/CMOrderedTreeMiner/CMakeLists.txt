add_library(cm_ordered_treeminer
OccLongList
OccList
TextTree
PatternTree.h
CMRmisc.h
)

add_executable(CMOrderedTreeMiner CMOrderedTreeMiner.cpp )

target_link_libraries(CMOrderedTreeMiner cm_ordered_treeminer)
