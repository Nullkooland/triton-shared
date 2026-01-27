#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

} // namespace mlir::triton
