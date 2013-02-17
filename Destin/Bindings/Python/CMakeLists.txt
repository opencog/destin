FIND_PACKAGE(PythonLibs)
    
if(PYTHONLIBS_FOUND)
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    SET_SOURCE_FILES_PROPERTIES(swig_destin_python.i PROPERTIES CPLUSPLUS ON)
        
    SET(CMAKE_SWIG_OUTDIR . )            
    SWIG_ADD_MODULE(pydestin python swig_destin_python.i )
    
    SWIG_LINK_LIBRARIES(pydestin ${highgui} destinalt common som cm_ordered_treeminer)
    
endif(PYTHONLIBS_FOUND)
