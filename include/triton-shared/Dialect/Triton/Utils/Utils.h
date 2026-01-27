#ifndef TRITON_SHARED_UTILITY_H
#define TRITON_SHARED_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

// Return true if the input type is a triton pointer or a tensor of triton
// pointers
bool isPtrTypeLike(Type t);

} // namespace mlir::triton

#endif // TRITON_SHARED_UTILITY_H
