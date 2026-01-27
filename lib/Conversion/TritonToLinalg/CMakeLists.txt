#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(TritonToLinalg
  TritonToLinalg.cpp
  ReconcilePtrCasts.cpp
  TritonToPtr.cpp
  CollapseShape.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  TritonSharedUtils

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
  TritonToUnstructured
  TritonPtrToMemref
  UnstructuredToMemref
)
