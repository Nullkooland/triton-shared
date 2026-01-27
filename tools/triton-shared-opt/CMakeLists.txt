get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(triton-shared-opt triton-shared-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-shared-opt)
target_link_libraries(triton-shared-opt PRIVATE
  TritonTransforms
  TritonSharedTransforms
  TritonToLinalg
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-shared-opt)
