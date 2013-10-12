FIND_PACKAGE(JNI)
if(JNI_FOUND)
    include_directories( ${JNI_INCLUDE_DIRS} )
    SET_SOURCE_FILES_PROPERTIES(swig_destin_java.i PROPERTIES CPLUSPLUS ON)
    SET_SOURCE_FILES_PROPERTIES(swig_destin_java.i PROPERTIES SWIG_FLAGS "-package;javadestin"  )
    SET(CMAKE_SWIG_OUTDIR ./gen_src/javadestin )            
    SWIG_ADD_MODULE(destinjava java swig_destin_java.i)
        
    SWIG_LINK_LIBRARIES(destinjava destinalt common som cm_ordered_treeminer ${OpenCV_LIBS})

    install(TARGETS destinjava
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
    )
    
else(JNI_FOUND)
    message("Couldn't find jni. Is a jdk installed? Not building java interfaces.")
endif(JNI_FOUND)

