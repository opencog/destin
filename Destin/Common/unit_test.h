#ifndef unit_test_h
#define unit_test_h

#include <stdbool.h>
#include <stdio.h>

#include <stdarg.h>
#include <stdlib.h>
#include <math.h>

int TEST_HAS_FAILURES = false;

#define RUN( f ){                           \
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

bool _assertFloatArrayEquals(float * expected, float * actual, int length, double epsilon, int line){
    int i;
    if(length <= 0 ){
        printf("assertFloatArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if(fabs( expected[i] - actual[i]) > epsilon ){
            printf("assertFloatArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %f, actual: %f\n, difference: %e", expected[i], actual[i], expected[i] - actual[i]);
            return false;
        }
    }
    return true;
}


#define assertFloatArrayEquals( exp, act, len ){\
    if( ! _assertFloatArrayEquals( (exp), (act), (len), 0.0, __LINE__ ) ){\
        return 1;\
    }\
}\

/** same as assertFloatArrayEquals but asserts true if difference is less than epsilon
 *
 */
#define assertFloatArrayEqualsE( exp, act, len, epsilon ){\
    if( ! _assertFloatArrayEquals( (exp), (act), (len), epsilon, __LINE__ ) ){\
        return 1;\
    }\
}\


void printFloatArray(float * array, int length){
    int i;
    printf("float array: ");
    for(i = 0; i < length - 1 ; i++){
        printf("%i: %f, ", i,array[i]);
    }
    printf("%i: %f\n",i,array[i]);
}


/** converts arguments into an array which is returned.
 *
 * argc: how many arguments passed
 * Be sure to write float constants. To pass 1 for example, write 1.0 or they
 * will be skipped or other errors may occur
 *
 * user takes ownership of array.
 */
float * toFloatArray(long c, ...) {

    //compiler note: ‘float’ is promoted to ‘double’ when passed through ‘...’
    //so have to start with double then cast to float
    float * a = (float *)malloc(c * sizeof(float));

    if(a==NULL){
        printf("toArray error\n");
        exit(1);
    }
    va_list arg_list;

    va_start(arg_list, c);
    int arg_count = 0;
    int i;
    for ( i = 0; i < c; i++ ){
        double d = va_arg(arg_list, double);
        a[i] = d;
    }
    va_end(arg_list);

    return a;
}

#endif
