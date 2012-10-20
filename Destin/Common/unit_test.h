#ifndef unit_test_h
#define unit_test_h

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

#define assertFloatEquals( expected, actual )\
    if( (expected) != (actual)  ){\
        printf("assertFloatEquals FAILED, line: %i, expected: %f, actual: %f\n", __LINE__, expected, actual);\
        return 1;\
    }\

bool _assertFloatArrayEquals(float * expected, float * actual, int length, int line){
    int i;
    if(length <= 0 ){
        printf("assertFloatArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if(expected[i] != actual[i]){
            printf("assertFloatArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %f, actual: %f\n", expected[i], actual[i]);
            return false;
        }
    }
    return true;
}


#define assertFloatArrayEquals( exp, act, len )\
if( ! _assertFloatArrayEquals( (exp), (act), (len), __LINE__ ) ){\
    return 1;\
}\


void printFloatArray(float * array, int length){
    int i;
    printf("float array: ");
    for(i = 0; i < length - 1 ; i++){
        printf("%i: %f, ", i,array[i]);
    }
    printf("%i: %f\n",i,array[i]);
}
#endif
