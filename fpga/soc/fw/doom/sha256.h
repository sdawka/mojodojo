#ifndef SOC_SHA256_H
#define SOC_SHA256_H
#include <stddef.h>
#include <stdint.h>
void sha256(const void *data, size_t len, uint8_t out[32]);
#endif
