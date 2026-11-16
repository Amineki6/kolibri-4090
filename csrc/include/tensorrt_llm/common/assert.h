#pragma once

#include <stdexcept>

#define TLLM_CHECK_WITH_INFO(cond, msg) \
    do {                                \
        if (!(cond)) {                  \
            throw std::runtime_error(msg); \
        }                               \
    } while (0)
