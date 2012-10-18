#ifndef test_h
#define test_h

#include <stdbool.h>
#include <stdio.h>


int TEST_HAS_FAILURES = false;

#define RUN( f )                            \
{                                           \
    printf("Running " #f "\n");             \
    int r = f();                            \
    if(r != 0){                             \
        TEST_HAS_FAILURES = true;           \
        printf("FAILED\n" );                \
    }else{                                  \
        printf("\n");                       \
    }                                       \
}                                           \

#define assertTrue( E )\
{\
    if(!( E )  ){\
        printf("assertTrue FAILED, line: %i, expression: " #E "\n", __LINE__);\
        return 1;\
    }\
}\

bool _assertFloatArrayEquals(float * expected, float * actual, int length){
    int i;
    for(i = 0 ; i < length ; i++){
        if(expected[i] != actual[i]){
            printf("assertFloatArrayEquals FAILED, line: %i, on index %i with array length %i\n", __LINE__, i, length );
            printf("expected: %f, actual: %f\n", expected[i], actual[i]);
            return false;
        }
    }
    return true;
}


#define assertFloatArrayEquals( exp, act, len )\
if( ! _assertFloatArrayEquals( (exp), (act), (len)) ){\
    return 1;\
}\

#endif
