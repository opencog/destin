#ifndef __ARRAY_H
#define __ARRAY_H

#include <string.h>
#include "macros.h"

/*
 * Insert element into array. The array length is increased by one.
 * 
 * Example:
 * Input:   array = {1 1 1 1 2 2}, length = 6, index = 4, value = 3
 * Result:  array = {1 1 1 1 3 2 2}
 */
void ArrayInsertElement(void **array, size_t size, uint length, uint index, void *value);

void ArrayInsertInt(int **array, uint length, uint index, int value);
void ArrayInsertUInt(uint **array, uint length, uint index, uint value);
void ArrayInsertLong(long **array, uint length, uint index, long value);
void ArrayInsertFloat(float **array, uint length, uint index, float value);
void ArrayInsertPtr(void **array, uint length, uint index, void *value);

#define ArrayAppendInt(array, length, value)    ArrayInsertInt(array, length, length, value)
#define ArrayAppendUInt(array, length, value)   ArrayInsertUInt(array, length, length, value)
#define ArrayAppendLong(array, length, value)   ArrayInsertLong(array, length, length, value)
#define ArrayAppendFloat(array, length, value)  ArrayInsertFloat(array, length, length, value)
#define ArrayAppendPtr(array, length, value)    ArrayInsertPtr(array, length, length, value)

/*
 * Insert multiple elements into the array. The array length is increased by a number of inserted elements (indexLength)
 * Index array must be sorted in ascending order.
 * 
 * Example:
 * Input:   array = {1 1 2 2 2 3 4 4}, length = 8, index = {2 5 6}, values = {7 8 9} indexLength = 3
 * Result:  array = {1 1 7 2 2 2 8 3 9 4 4}
 */
void ArrayInsertMultiple(void **array, size_t size, uint length, uint *index, void *values, uint indexLength);

void ArrayInsertInts(int **array, uint length, uint *index, int *values, uint indexLength);
void ArrayInsertUInts(uint **array, uint length, uint *index, uint *values, uint indexLength);
void ArrayInsertLongs(long **array, uint length, uint *index, long *values, uint indexLength);
void ArrayInsertFloats(float **array, uint length, uint *index, float *values, uint indexLength);
void ArrayInsertPtrs(void **array, uint length, uint *index, void *values, uint indexLength);


/*
 * Delete element from array. The array length is decresed by one.
 * 
 * Example:
 * Input:   array = {1 1 1 1 2 2 2}, length = 7, index = 4
 * Result:  array = {1 1 1 1 2 2}
 *
 * @param DeleteElement - pointer to destructor of deleted element. if DeleteElement is null then it is not called
 */
void ArrayDeleteElement(void **array, size_t size, uint length, uint index, void (*DeleteElement)(void *));
// Default destructor for ArrayDeleteArray
void ArrayFreeElement(void * element);

void ArrayDeleteInt(int **array, uint length, uint index);
void ArrayDeleteUInt(uint **array, uint length, uint index);
void ArrayDeleteLong(long **array, uint length, uint index);
void ArrayDeleteFloat(float **array, uint length, uint index);
void ArrayDeletePtr(void **array, uint length, uint index);
void ArrayDeleteArray(void **array, uint length, uint index);


/*
 * Delete multiple elements from array. The array length is increased by a number of deleted elements (indexLength)
 * Index array must be sorted in ascending order.
 * 
 * Example:
 * Input:  array = {1 1 7 2 2 2 8 3 9 4 4}, length = 11, index = {2 6 8}, indexLength=3
 * Result: array = {1 1 2 2 2 3 4 4}
 *
 * @param DeleteElement - pointer to destructor of deleted element. if DeleteElement is null then it is not called
 */
void ArrayDeleteMultiple(void **array, size_t size, uint length, uint *index, uint indexLength, void (*DeleteElement)(void *));

void ArrayDeleteInts(int **array, uint length, uint *index, uint indexLength);
void ArrayDeleteUInts(uint **array, uint length, uint *index, uint indexLength);
void ArrayDeleteLongs(long **array, uint length, uint *index, uint indexLength);
void ArrayDeleteFloats(float **array, uint length, uint *index, uint indexLength);
void ArrayDeletePtrs(void **array, uint length, uint *index, uint indexLength);
void ArrayDeleteArrays(void **array, uint length, uint *index, uint indexLength);


#endif