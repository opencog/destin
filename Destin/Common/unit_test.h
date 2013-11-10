#ifndef unit_test_h
#define unit_test_h

#include <stdbool.h>
#include <stdio.h>

#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <sys/time.h>

#ifdef _WIN32
typedef unsigned int uint;
#endif
int TEST_HAS_FAILURES = false; //checked at the end to determine if any tests have failed

#define ut_oops(s) { fprintf(stderr, s); exit(1); }

#define UT_MALLOC(s,t,n) {                              \
    if((s = (t *) malloc(n*sizeof(t))) == NULL) {       \
        ut_oops("error: malloc()\n");                   \
    }                                                   \
}

/**
 * Runs the test function passed in by name.
 * The test function should return 0 for passing or non zero otherwise.
 */
#define RUN( f ){                           \
    printf("**Running " #f "\n");             \
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
    if(!( (E) == true )  ){\
        printf("assertTrue FAILED, line: %i, expression: " #E "\n", __LINE__);\
        return 1;\
    }\
}\

#define assertFalse( E)\
{\
    if( ( (E) != false )  ){\
        printf("assertFalse FAILED, line: %i, expression: " #E "\n", __LINE__);\
        return 1;\
    }\
}\

#define assertFloatEquals( expected, actual, epsilon ){\
    if( isnan(expected) || isnan(actual) || fabs((expected) - (actual) ) > epsilon  ){\
        printf("assertFloatEquals FAILED, line: %i, expected: %f, actual: %f, difference: %e\n", __LINE__, (expected), (actual), ((expected) - (actual)));\
        return 1;\
    }\
}\

bool _assertFloatArrayEquals(float * expected, float * actual, int length, double epsilon, int line){
    int i;
    if(length <= 0 ){
        printf("assertFloatArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if(isnan(expected[i]) || isnan(actual[i]) || fabs( expected[i] - actual[i]) > epsilon ){
            printf("assertFloatArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %e, actual: %e, difference: %e\n", expected[i], actual[i], expected[i] - actual[i]);
            return false;
        }
    }
    return true;
}

bool _assertFloatArrayEqualsE2D(float ** expected, float ** actual, int rows, int cols, double epsilon, int line){
    int i, j;
    if(rows <= 0 ){
        printf("assertFloatArrayEquals2D FAILED, line: %i, negative or zero array rows: %i", line, rows);
        return false;
    }
    if(cols <= 0 ){
        printf("assertFloatArrayEquals2D FAILED, line: %i, negative or zero array columns: %i", line, cols);
        return false;
    }

    for(i = 0 ; i < rows ; i++){
        for(j = 0 ; j < cols; j++){
            if(isnan(expected[i][j]) || isnan(actual[i][j]) || fabs( expected[i][j] - actual[i][j]) > epsilon ){
                printf("assertFloatArrayEquals2D FAILED, line: %i, on (%i, %i) with array size (%i, %i)\n",
                       line, i, j, rows, cols);
                printf("expected: %e, actual: %e, difference: %e\n", expected[i][j], actual[i][j], expected[i][j] - actual[i][j]);
                return false;
            }
        }
    }
    return true;
}


bool _assertIntArrayEquals(int * expected, int * actual, int length, int line){
    int i;
    if(length <= 0 ){
        printf("assertIntArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if( expected[i] != actual[i] ){
            printf("assertIntArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %i, actual: %i\n", expected[i], actual[i]);
            return false;
        }
    }
    return true;
}


bool _assertShortArrayEquals(short * expected, short * actual, int length, int line){
    int i;
    if(length <= 0 ){
        printf("assertShortArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if( expected[i] != actual[i] ){
            printf("assertShortArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %i, actual: %i\n", expected[i], actual[i]);
            return false;
        }
    }
    return true;
}


bool _assertUIntArrayEquals(unsigned int * expected, unsigned int * actual, int length, int line){
    int i;
    if(length <= 0 ){
        printf("assertUIntArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if( expected[i] != actual[i] ){
            printf("assertUIntArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %i, actual: %i\n", expected[i], actual[i]);
            return false;
        }
    }
    return true;
}


bool _assertLongArrayEquals(long * expected, long * actual, int length, int line){
    int i;
    if(length <= 0 ){
        printf("assertLongArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if( expected[i] != actual[i] ){
            printf("assertLongArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %li, actual: %li\n", expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

bool _assertBoolArrayEquals(bool * expected, bool * actual, int length, int line){
    int i;
    if(length <= 0 ){
        printf("assertBoolArrayEquals FAILED, line: %i, negative or zero array length: %i", line, length);
        return false;
    }

    for(i = 0 ; i < length ; i++){
        if( expected[i] != actual[i] ){
            printf("assertBoolArrayEquals FAILED, line: %i, on index %i with array length %i\n", line, i, length );
            printf("expected: %s, actual: %s\n", expected[i] ? "true" : "false", actual[i] ? "true": "false");
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

#define assertShortArrayEquals( exp, act, len ){\
    if( ! _assertShortArrayEquals((exp), (act), (len), __LINE__ ) ){\
        return 1;\
    }\
}\


/** Fills dest array with caller's of this macro recieved arguments
 * last_fixed_argument - plain text of the last fixed argument passed to
 * the caller function, which will not be put into dest array
 * length - length of the arguments array
 */
#define ut_varags_to_array( dest, last_fixed_argument, length, the_type ){\
    va_list arg_list;\
    va_start(arg_list, last_fixed_argument);\
    int i;\
    for ( i = 0; i < length; i++ ){\
        dest[i] = va_arg(arg_list, the_type);\
    }\
    va_end(arg_list);\
}\

#define ut_varags_to_array2d( dest, last_fixed_argument, rows, cols, the_type ){\
    va_list arg_list;\
    va_start(arg_list, last_fixed_argument);\
    int i, j;\
    for ( i = 0; i < rows; i++ ){\
        for ( j = 0; j < cols; j++ ){\
            dest[i][j] = va_arg(arg_list, the_type);\
        }\
    }\
    va_end(arg_list);\
}\

#define ut_flatten_2d_array( dest, src, rows, cols ){\
    int i, j;\
    for ( i = 0; i < rows; i++ ){\
        for ( j = 0; j < cols; j++ ){\
            dest[i*cols + j] = src[i][j];\
        }\
    }\
}\

/** same as assertFloatArrayEquals but asserts true if difference is less than epsilon
 */
#define assertFloatArrayEqualsE( exp, act, len, epsilon ){\
    if( ! _assertFloatArrayEquals( (exp), (act), (len), epsilon, __LINE__ ) ){\
        return 1;\
    }\
}\

bool _assertFloatArrayEqualsEV(float *actual, float epsilon, int len, int line, ...){
    float expected[len];
    ut_varags_to_array(expected, line, len, double);
    return _assertFloatArrayEquals(expected, actual, len, epsilon, line );
}

bool _assertFloatArrayEqualsE2DV(float **actual2D, float epsilon, int rows, int cols, int line, ...){
    int len = rows * cols;
    float expected[len];
    float actual[len];
    ut_varags_to_array(expected, line, len, double);
    ut_flatten_2d_array(actual, actual2D, rows, cols);
    return _assertFloatArrayEquals(expected, actual, len, epsilon, line );
}

/** Test float array with epislon and variable arguments.
  * Must write floats as 1.0 and not 1 for example or
  * bad things will happen.
  */
#define assertFloatArrayEqualsEV( act, epsilon, len, args... ){\
    if(! _assertFloatArrayEqualsEV(act, epsilon, len, __LINE__, args )){\
        return 1;\
    }\
}\

#define assertFloatArrayEqualsE2DV( act, epsilon, rows, cols, args... ){\
    if(! _assertFloatArrayEqualsE2DV(act, epsilon, rows, cols, __LINE__, args )){\
        return 1;\
    }\
}\

#define assertFloatArrayEqualsE2D( exp, act, rows, cols, epsilon){\
    if(! _assertFloatArrayEqualsE2D(exp, act, rows, cols, epsilon, __LINE__)){\
        return 1;\
    }\
}\

bool _assertIntArrayEqualsV(int *actual, int len, int line, ...){
    int expected[len];
    ut_varags_to_array(expected, line, len, int);
    return _assertIntArrayEquals(expected, actual, len, line );
}

bool _assertShortArrayEqualsV(short *actual, int len, int line, ...){
    short expected[len];
    ut_varags_to_array(expected, line, len, int); //short is promoted to int
    return _assertShortArrayEquals(expected, actual, len, line );
}

bool _assertUIntArrayEqualsV(unsigned int *actual, int len, int line, ...){
    unsigned int expected[len];
    ut_varags_to_array(expected, line, len, unsigned int);
    return _assertUIntArrayEquals(expected, actual, len, line );
}

bool _assertLongArrayEqualsV(long *actual, int len, int line, ...){
    long expected[len];
    ut_varags_to_array(expected, line, len, long);
    return _assertLongArrayEquals(expected, actual, len, line );
}

bool _assertBoolArrayEqualsV(bool *actual, int len, int line, ...){
    bool expected[len];
    ut_varags_to_array(expected, line, len, int); //use int because bool is promoted to int by compiler
    return _assertBoolArrayEquals(expected, actual, len, line );
}

/** test short array with variable arguments
*/
#define assertShortArrayEqualsV( act, len, expecteds... ){\
    if(! _assertShortArrayEqualsV((act), (len), __LINE__, expecteds )){\
        return 1;\
    }\
}\

/** test int array with variable arguments
*/
#define assertIntArrayEqualsV( act, len, expecteds... ){\
    if(! _assertIntArrayEqualsV(act, len, __LINE__, expecteds )){\
        return 1;\
    }\
}\

/** test int array with variable arguments
*/
#define assertUIntArrayEqualsV( act, len, expecteds... ){\
    if(! _assertUIntArrayEqualsV(act, len, __LINE__, expecteds )){\
        return 1;\
    }\
}\


/** Test long array with variable arguments
* MUST write long constants as 1L, 2L ect. or bad things will happen.
*/
#define assertLongArrayEqualsV( act, len, expecteds... ){\
    if(! _assertLongArrayEqualsV(act, len, __LINE__, expecteds )){\
        return 1;\
    }\
}\

/** test boolean array with variable arguments
*/
#define assertBoolArrayEqualsV( act, len, expecteds... ){\
    if(! _assertBoolArrayEqualsV(act, len, __LINE__, expecteds )){\
        return 1;\
    }\
}\


// macro to generate print array functions
#define UT_PRINT_ARRAY(the_type, format_flag)\
    void print_##the_type##_array(the_type * array, int length){\
        int i;\
        printf(#the_type " array:");\
        for(i = 0 ; i < length -1 ; i++){\
            printf(" %i: %" format_flag ", ", i, array[i]);\
        }\
        printf("%i: %" format_flag "\n", i, array[i]);\
    }\

// define print array functions for each type
UT_PRINT_ARRAY(int, "i")
UT_PRINT_ARRAY(uint, "u")
UT_PRINT_ARRAY(float, "e")
UT_PRINT_ARRAY(long, "li")

void printFloatArray(float * array, int length){
    int i;
    printf("float array: ");
    for(i = 0; i < length - 1 ; i++){
        printf("%i: %e, ", i,array[i]);
    }
    printf("%i: %e\n",i,array[i]);
}


/** assigns the dest array the float values pass as arguments
 * CAUTION: Be sure to write float constants. To pass 1 for example, write 1.0
 * and not 1 or they will be skipped or other errors may occur
 *
 * @length - number of float elements passed in.
 */
void assignFloatArray(float * dest, int length, ...){

    ut_varags_to_array(dest, length, length, double);
}

/** assigns the dest array the float values pass as arguments
 * CAUTION: Be sure to write float constants. To pass 1 for example, write 1.0
 * and not 1 or they will be skipped or other errors may occur
 * dest array has 2 dimensions
 *
 * @rows - number of rows
 * @cols - number of columns
 */
void assignFloatArray2D(float ** dest, int rows, int cols, ...){

    ut_varags_to_array2d(dest, cols, rows, cols, double);
}

/** assigns the dest unsigned int array with the int values pass as arguments
 * CAUTION: Be sure to write int constants. To pass 1 for example, write 1
 * and not 1.0 or they will be skipped or other errors may occur
 *
 * @length - number of int elements passed in.
 */
void assignUIntArray(uint * dest, int length, ...){

    ut_varags_to_array(dest, length, length, int);
}



/** converts arguments into an array which is returned.
 *
 * CAUTION: Be sure to write float constants. To pass 1 for example, write 1.0
 * and not 1 or they will be skipped or other errors may occur
 *
 * @param c: how many arguments passed
 *
 * @return - newly created array, user is responsible for freeing it.
 */
float * toFloatArray(long c, ...) {

    //compiler note: ‘float’ is promoted to ‘double’ when passed through ‘...’
    //so have to start with double then cast to float
    float * a = (float *)malloc(c * sizeof(float));

    if(a==NULL){
        printf("toArray error\n");
        exit(1);
    }

    ut_varags_to_array(a, c, c, double);

    return a;
}

bool _assertIntEquals( int expected, int actual, int line){
    if( expected == actual ){
        return true;
    }else{
        printf("assertIntEquals FAILED, line %i, expected: %i, actual: %i\n", line, expected, actual );
        return false;
    }

}

#define assertIntEquals( expected, actual){\
    if( ! _assertIntEquals((expected), (actual), __LINE__ )){\
        return 1;\
    }\
}\




/** Copies and returns a 2 dimenional array.
* @param src - source 2 dimensional array to copy
* @param rows - how many child arrays in the parent array, i.e. size of first dimension
* @param cols - size of child arrays, i.e size of second dimension
* @return - copied array, user is responsible for freeing it when no longer needed.
*/
float** copyFloatDim2Array(float** src, int rows, int cols){
   float ** ret;
   int r;
   UT_MALLOC(ret, float *, rows);
    for(r = 0 ; r < rows ; r++){
        UT_MALLOC(ret[r], float, cols );
        memcpy(ret[r], src[r], sizeof(float) * cols );
    }
    return ret;
}

bool _assertNoNans(float * array, int length, int line){
    int i;
    for(i = 0 ; i < length ; i++){
        if(isnan(array[i])){
            printf("assertNoNans FAILED, line %i, index %i of %i\n", line, i, length);
            return false;
        }
    }
    return true;
}

/** assertNoNans - asserts that none of the elements of the float array contain nans
 */
#define assertNoNans(float_array, len){\
    if(! _assertNoNans(float_array, len, __LINE__)){\
        return 1;\
    }\
}\

/** force a failure
  */
#define testFailed(message){\
    printf("test FAILED, line %i, message: %s", __LINE__, #message "\n" );\
    return 1;\
}\

/** Call this at the end to report test results
  */
#define UT_REPORT_RESULTS(){\
printf("FINSHED TESTING: %s\n", TEST_HAS_FAILURES ? "FAIL" : "PASS");\
if(TEST_HAS_FAILURES){\
    return 1;\
}\
}\

/**
 * A simple random number generator for testing that should return the
 * same sequence of numbers across platforms / compilers.
 * Found from http://pubs.opengroup.org/onlinepubs/009695399/functions/rand.html
 */
static unsigned long dst_ut_rand_next = 1;
#define DST_UT_RAND_MAX 32767
int dst_ut_int_rand(){  /* RAND_MAX assumed to be 32767. */
    dst_ut_rand_next = dst_ut_rand_next * 1103515245 + 12345;
    return((unsigned)(dst_ut_rand_next/65536) % (DST_UT_RAND_MAX + 1));
}
// generate a float between 0.0 and 1.0
float dst_ut_float_rand(){
    return (float)dst_ut_int_rand() / (float)DST_UT_RAND_MAX;
}
// seed the random number generator
void dst_ut_srand(unsigned long seed) {
    dst_ut_rand_next = seed;
}
// seed the random number generator with milliseconds since epoch
void dst_ut_srand_milliseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long millisecondsSinceEpoch =
        (unsigned long long)(tv.tv_sec) * 1000 +
        (unsigned long long)(tv.tv_usec) / 1000;

    dst_ut_rand_next = (unsigned long)millisecondsSinceEpoch;
}

/**
 * Makes a double array of random floats between 0 and 1.
 * Should be freed with freeRandomImages.
 * Uses dst_ut_float_rand() to generate the values.
 * @param image_size - total number of pixels per image
 * @param nImages - number of images to create
 * @param seed - seed for random number generator
 * @return the generated images.
 */
float ** makeRandomImages(uint image_size, uint nImages){
    //generate random images
    uint i, j;
    float ** images;
    images = (float **)malloc(sizeof(float *) * nImages);
    if(images == NULL){
        ut_oops("makeRandomImages: could not malloc\n");
    }

    for(i = 0 ; i < nImages; i++){
        images[i] = (float *)malloc(sizeof(float) * image_size);
        if(images[i] == NULL){
            ut_oops("makeRandomImages: could not malloc\n");
        }
    }

    for(i = 0 ; i < image_size ; i++){
        for(j = 0 ; j < nImages ; j++){
            images[j][i] = dst_ut_float_rand();
        }
    }
    return images;
}

/**
 * Frees the images created from makeRandomImages function.
 * @param images - pointer to the images to be freed.
 * @param nImages - the number of images
 */
void freeRandomImages(float ** images, uint nImages){
    uint i;
    if(images == NULL){
        ut_oops("Trying to free null pointer in freeRandomImages.\n");
    }
    for(i = 0 ; i < nImages ; i++){
        if(images[i] == NULL){
            ut_oops("Trying to free null pointer in freeRandomImages.\n")
        }
        free(images[i]);
    }
    free(images);
    return;
}

#endif
