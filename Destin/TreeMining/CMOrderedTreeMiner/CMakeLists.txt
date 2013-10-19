add_library(cm_ordered_treeminer SHARED
OccLongList
OccList
TextTree
PatternTree.h
CMRmisc.h
)

add_executable(CMOrderedTreeMiner CMOrderedTreeMiner.cpp )

target_link_libraries(CMOrderedTreeMiner cm_ordered_treeminer)

install(TARGETS cm_ordered_treeminer CMOrderedTreeMiner
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static )
