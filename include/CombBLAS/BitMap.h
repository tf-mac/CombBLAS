#ifndef COMBBLAS_BITMAP_H
#define COMBBLAS_BITMAP_H

#include <stdint.h>

#include <algorithm>
#include <iostream>

#define WORD_OFFSET(n) (n / 64)
#define BIT_OFFSET(n) (n & 0x3f)

namespace combblas {
class BitMap {
public:
    // Constructors
    BitMap();

    BitMap(uint64_t size);

    // Destructor
    ~BitMap();

    // Copy constructor and assignment operator
    BitMap(const BitMap &rhs);

    BitMap &operator=(const BitMap &rhs);

    // Member functions
    void reset();

    void set_bit(uint64_t pos);

    void reset_bit(uint64_t pos);

    void set_bit_atomic(long pos);

    bool get_bit(uint64_t pos);

    long get_next_bit(uint64_t pos);

    uint64_t *data();

    void copy_from(const BitMap *other);

    void print_ones();

private:
    uint64_t *start;
    uint64_t *end;
};
} // namespace combblas

// Include the source file at the end of the header.
// #include "Bitmap.cpp"

#endif  // BITMAP_H
