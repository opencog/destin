#include <stdio.h>
#include <math.h>

#include "macros.h"
#include "array.h"

void ArrayInsertElement(void **array, size_t size, uint length, uint index, void *value)
{
    char * _array = (char *) *array;
    if (index > length)
    {
        fprintf(stderr, "ArrayInsertElement(): index is out of range!\n");
        return;
    }

    REALLOCV(_array, char, (length * size), size);
    *array = (void *) _array;

    if (index < length)
    {
        memmove(_array + (index + 1) * size, _array + index * size , (length - index) * size);
    }
    memcpy(_array + index * size, value, size);
}

void ArrayInsertInt(int **array, uint length, uint index, int value)
{
    ArrayInsertElement((void *) array, sizeof(int), length, index, &value);
}

void ArrayInsertUInt(uint **array, uint length, uint index, uint value)
{
    ArrayInsertElement((void *) array, sizeof(uint), length, index, &value);
}

void ArrayInsertLong(long **array, uint length, uint index, long value)
{
    ArrayInsertElement((void *) array, sizeof(long), length, index, &value);
}

void ArrayInsertFloat(float **array, uint length, uint index, float value)
{
    ArrayInsertElement((void *) array, sizeof(float), length, index, &value);
}

void ArrayInsertPtr(void **array, uint length, uint index, void *value)
{
    ArrayInsertElement(array, sizeof(void *), length, index, &value);
}


void ArrayInsertMultiple(void **array, size_t size, uint length, uint *index, void *values, uint indexLength)
{
    if (indexLength == 0)
        return;
    if (index[indexLength - 1] > length)
    {
        fprintf(stderr, "ArrayInsertMultiple(): index is out of range!\n");
        return;
    }

    char * _array = (char *) *array;
    char * _values = (char *) values;
    REALLOCV(_array, char, (length * size), (indexLength * size));
    *array = (void *) _array;

    uint lastIndex = length;
    int k;
    for (k = indexLength - 1; k >= 0; k--)
    {
        if (index[k] > lastIndex)
        {
            fprintf(stderr, "ArrayInsertMultiple(): wrong order of indexes!\n");
            return;
        }
        if (index[k] < lastIndex)
        {
            memmove(_array + (index[k] + k + 1) * size, _array + index[k] * size, (lastIndex - index[k]) * size);
        }
        memcpy (_array + (index[k] + k) * size, _values + k * size, size);
        lastIndex = index[k];
    }
}

void ArrayInsertInts(int **array, uint length, uint *index, int *values, uint indexLength)
{
    ArrayInsertMultiple((void *) array, sizeof(int), length, index, values, indexLength);
}

void ArrayInsertUInts(uint **array, uint length, uint *index, uint *values, uint indexLength)
{
    ArrayInsertMultiple((void *) array, sizeof(uint), length, index, values, indexLength);
}

void ArrayInsertLongs(long **array, uint length, uint *index, long *values, uint indexLength)
{
    ArrayInsertMultiple((void *) array, sizeof(long), length, index, values, indexLength);
}

void ArrayInsertFloats(float **array, uint length, uint *index, float *values, uint indexLength)
{
    ArrayInsertMultiple((void *) array, sizeof(float), length, index, values, indexLength);
}

void ArrayInsertPtrs(void **array, uint length, uint *index, void *values, uint indexLength)
{
    ArrayInsertMultiple((void *) array, sizeof(void *), length, index, values, indexLength);
}


void ArrayDeleteElement(void **array, size_t size, uint length, uint index, void (*DeleteElement)(void *))
{
    if (index >= length)
    {
        fprintf(stderr, "ArrayDeleteElement(): index is out of range!\n");
        return;
    }

    char *_array = (char *) *array;
    if (DeleteElement != NULL)
    {
        DeleteElement((void *) _array + index * size);
    }
    if (index < length - 1)
    {
        memmove(_array + index * size, _array + (index + 1) * size, (length - index - 1) * size);
    }
}

void ArrayFreeElement(void * element)
{
    char ** _element = (char **) element;
    FREE(*_element);
}

void ArrayDeleteInt(int **array, uint length, uint index)
{
    ArrayDeleteElement((void *) array, sizeof(int), length, index, NULL);
}

void ArrayDeleteUInt(uint **array, uint length, uint index)
{
    ArrayDeleteElement((void *) array, sizeof(uint), length, index, NULL);
}

void ArrayDeleteLong(long **array, uint length, uint index)
{
    ArrayDeleteElement((void *) array, sizeof(long), length, index, NULL);
}

void ArrayDeleteFloat(float **array, uint length, uint index)
{
    ArrayDeleteElement((void *) array, sizeof(float), length, index, NULL);
}

void ArrayDeletePtr(void **array, uint length, uint index)
{
    ArrayDeleteElement((void *) array, sizeof(void *), length, index, NULL);
}

void ArrayDeleteArray(void **array, uint length, uint index)
{
    ArrayDeleteElement((void *) array, sizeof(void *), length, index, &ArrayFreeElement);
}


void ArrayDeleteMultiple(void **array, size_t size, uint length, uint *index, uint indexLength, void (*DeleteElement)(void *))
{
    if (indexLength == 0)
        return;
    uint k;
    for (k = 0; k < indexLength; k++)
    {
        if (index[k] >= length)
        {
            fprintf(stderr, "ArrayDeleteMultiple(): index is out of range!\n");
            return;
        }
    }

    char * _array = (char *) *array;
    for (k = 0; k < indexLength - 1; k++)
    {
        if (index[k] >= index[k + 1])
        {
            fprintf(stderr, "ArrayDeleteMultiple(): wrong order of indexes!\n");
            return;
        }
        if (DeleteElement != NULL)
        {
            DeleteElement((void *) _array + index[k] * size);
        }
        if (index[k] < index[k + 1] - 1)
        {
            memmove(_array + (index[k] - k) * size, _array + (index[k] + 1) * size, (index[k+1] - index[k] - 1) * size);
        }
    }

    k = indexLength - 1;
    if (DeleteElement != NULL)
    {
        DeleteElement((void *) _array + index[k] * size);
    }
    if (index[k] < length - 1)
    {
        memmove(_array + (index[k] - k) * size, _array + (index[k] + 1) * size, (length - index[k] - 1) * size);
    }
}

void ArrayDeleteInts(int **array, uint length, uint *index, uint indexLength)
{
    ArrayDeleteMultiple((void *) array, sizeof(int), length, index, indexLength, NULL);
}

void ArrayDeleteUInts(uint **array, uint length, uint *index, uint indexLength)
{
    ArrayDeleteMultiple((void *) array, sizeof(uint), length, index, indexLength, NULL);
}

void ArrayDeleteLongs(long **array, uint length, uint *index, uint indexLength)
{
    ArrayDeleteMultiple((void *) array, sizeof(long), length, index, indexLength, NULL);
}

void ArrayDeleteFloats(float **array, uint length, uint *index, uint indexLength)
{
    ArrayDeleteMultiple((void *) array, sizeof(float), length, index, indexLength, NULL);
}

void ArrayDeletePtrs(void **array, uint length, uint *index, uint indexLength)
{
    ArrayDeleteMultiple((void *) array, sizeof(void *), length, index, indexLength, NULL);
}

void ArrayDeleteArrays(void **array, uint length, uint *index, uint indexLength)
{
    ArrayDeleteMultiple((void *) array, sizeof(void *), length, index, indexLength, &ArrayFreeElement);
}
