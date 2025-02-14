#include "CombBLAS/BitMap.h"

namespace combblas
{

// Default constructor
BitMap::BitMap() : start(nullptr), end(nullptr) {}

// Constructor with size parameter
BitMap::BitMap(uint64_t size)
{
    uint64_t num_longs = (size + 63) / 64;
    start = new uint64_t[num_longs]();  // Zero-initialized by default
    end = start + num_longs;
}

// Destructor
BitMap::~BitMap() { delete[] start; }

// Copy constructor
BitMap::BitMap(const BitMap &rhs)
{
    uint64_t num_longs = rhs.end - rhs.start;
    start = new uint64_t[num_longs];
    end = start + num_longs;
    std::copy(rhs.start, rhs.end, start);
}

// Assignment operator
BitMap &
BitMap::operator=(const BitMap &rhs)
{
    if (this != &rhs) {
        delete[] start;
        uint64_t num_longs = rhs.end - rhs.start;
        start = new uint64_t[num_longs];
        end = start + num_longs;
        std::copy(rhs.start, rhs.end, start);
    }
    return *this;
}

// Reset all bits to 0
void
BitMap::reset()
{
    for (uint64_t *it = start; it != end; ++it) {
        *it = 0;
    }
}

// Set a specific bit
void
BitMap::set_bit(uint64_t pos)
{
    start[WORD_OFFSET(pos)] |= (static_cast<uint64_t>(1l) << BIT_OFFSET(pos));
}

// Reset a specific bit
void
BitMap::reset_bit(uint64_t pos)
{
    start[WORD_OFFSET(pos)] &= ~(static_cast<uint64_t>(1l) << BIT_OFFSET(pos));
}

// Set a specific bit atomically (non-atomic version used here)
void
BitMap::set_bit_atomic(long pos)
{
    set_bit(pos);
    // For a true atomic implementation, one might use __sync_bool_compare_and_swap:
    // uint64_t old_val, new_val;
    // uint64_t *loc = start + WORD_OFFSET(pos);
    // do {
    //     old_val = *loc;
    //     new_val = old_val | (static_cast<uint64_t>(1l) << BIT_OFFSET(pos));
    // } while (!__sync_bool_compare_and_swap(loc, old_val, new_val));
}

// Get the value of a specific bit
bool
BitMap::get_bit(uint64_t pos)
{
    return (start[WORD_OFFSET(pos)] & (static_cast<uint64_t>(1l) << BIT_OFFSET(pos))) != 0;
}

// Get the next set bit after the given position
long
BitMap::get_next_bit(uint64_t pos)
{
    uint64_t next = pos;
    int bit_offset = BIT_OFFSET(pos);
    uint64_t *it = start + WORD_OFFSET(pos);
    uint64_t temp = *it;
    if (bit_offset != 63) {
        temp = temp >> (bit_offset + 1);
    } else {
        temp = 0;
    }
    if (!temp) {
        next = (next & 0xffffffc0);
        while (!temp) {
            ++it;
            if (it >= end) return -1;
            temp = *it;
            next += 64;
        }
    } else {
        next++;
    }
    while (!(temp & 1)) {
        temp = temp >> 1;
        next++;
    }
    return next;
}

// Return the underlying data pointer
uint64_t *
BitMap::data()
{
    return start;
}

// Copy data from another BitMap
void
BitMap::copy_from(const BitMap *other)
{
    std::copy(other->start, other->end, start);
}

// Print positions of all bits set to 1
void
BitMap::print_ones()
{
    uint64_t max_size = (end - start) * 64;
    for (uint64_t i = 0; i < max_size; i++) {
        if (get_bit(i)) {
            std::cerr << " " << i << std::endl;
        }
    }
}

}  // namespace combblas
