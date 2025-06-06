// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @wrap_stacked_masked_loop_01234567(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant dense<-9.900000e+01> : tensor<4x4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<3> : tensor<1x4xi32>
    %cst_1 = arith.constant dense<3> : tensor<4xi32>
    %cst_2 = arith.constant dense<2> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = arith.addi %0, %cst_2 : tensor<4xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %3 = arith.remsi %1, %2 : tensor<4xi32>
    %4 = arith.addi %0, %cst_1 : tensor<4xi32>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<4x1xi32>
    %7 = arith.muli %5, %6 : tensor<4x1xi32>
    %8 = tt.expand_dims %4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %9 = tt.splat %arg5 : i32 -> tensor<1x4xi32>
    %10 = arith.muli %8, %9 : tensor<1x4xi32>
    %11 = tt.broadcast %7 : tensor<4x1xi32> -> tensor<4x4xi32>
    %12 = tt.broadcast %10 : tensor<1x4xi32> -> tensor<4x4xi32>
    %13 = arith.addi %11, %12 : tensor<4x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %16 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %17 = tt.splat %arg6 : i32 -> tensor<4x1xi32>
    %18 = arith.muli %17, %16 : tensor<4x1xi32>
    %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %20 = tt.addptr %19, %18 : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.splat %arg7 : i32 -> tensor<1x4xi32>
    %23 = arith.muli %22, %21 : tensor<1x4xi32>
    %24 = tt.broadcast %20 : tensor<4x1x!tt.ptr<f32>> -> tensor<4x4x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x4xi32> -> tensor<4x4xi32>
    %26 = tt.addptr %24, %25 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %27 = arith.cmpi slt, %21, %cst_0 : tensor<1x4xi32>
    %28 = tt.broadcast %27 : tensor<1x4xi1> -> tensor<4x4xi1>
    %29 = arith.muli %arg5, %c4_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<4x4xi32>
    %31:2 = scf.for %arg8 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg9 = %15, %arg10 = %26) -> (tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>)  : i32 {
      %32 = tt.load %arg9, %28, %cst : tensor<4x4x!tt.ptr<f32>>
      tt.store %arg10, %32 : tensor<4x4x!tt.ptr<f32>>
      %33 = tt.addptr %arg9, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %34 = tt.addptr %arg10, %30 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      scf.yield %33, %34 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @wrap_stacked_masked_loop_01234567
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32) {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_minus_9_dot_900000_:%.+]] = arith.constant -9.900000e+01 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_2_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_0_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.muli [[PARAM_5_]], [[CST_4_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i32 to index
// CHECK-DAG:       [[VAR_10_:%.+]]:2 = scf.for [[VAR_arg14_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg15_:%.+]] = [[VAR_2_]], [[VAR_arg16_:%.+]] = [[CST_0_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg16_]]{{.}}, sizes: [4, 4], strides: {{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<*xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_arg15_]], [[VAR_5_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.remsi [[VAR_11_]], [[VAR_1_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.addi [[VAR_3_]], [[VAR_12_]] : index
// CHECK:             [[VAR_14_:%.+]] = arith.subi [[VAR_13_]], [[VAR_11_]] : index
// CHECK:             [[VAR_15_:%.+]] = arith.divsi [[VAR_14_]], [[VAR_1_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_11_]]{{.}}, sizes: {{.}}[[VAR_15_]], [[CST_4_]]{{.}}, strides: {{.}}[[VAR_1_]], [[VAR_4_]]{{.}} : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.subi [[CST_4_]], [[VAR_15_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_12_]]{{.}}, sizes: {{.}}[[VAR_16_]], [[CST_4_]]{{.}}, strides: {{.}}[[VAR_1_]], [[VAR_4_]]{{.}} : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<4x4xf32>
// CHECK:             linalg.fill ins([[CST_minus_9_dot_900000_]] : f32) outs([[RES_]] : memref<4x4xf32>)
// CHECK:             [[VAR_17_:%.+]] = arith.minsi [[VAR_15_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.subi [[CST_4_]], [[VAR_17_]] : index
// CHECK-DAG:         [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_0_]][0, 0] {{.}}[[VAR_17_]], 3] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_subview_2_:%.+]] = memref.subview [[VAR_reinterpret_cast_1_]][0, 0] {{.}}[[VAR_18_]], 3] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_subview_3_:%.+]] = memref.subview [[RES_]][0, 0] {{.}}[[VAR_17_]], 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1]>>
// CHECK-DAG:         [[VAR_subview_4_:%.+]] = memref.subview [[RES_]]{{.}}[[VAR_17_]], 0] {{.}}[[VAR_18_]], 3] [1, 1] : memref<4x4xf32> to memref<?x3xf32, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy [[VAR_subview_]], [[VAR_subview_3_]] : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1]>>
// CHECK:             memref.copy [[VAR_subview_2_]], [[VAR_subview_4_]] : memref<?x3xf32, strided<[?, ?], offset: ?>> to memref<?x3xf32, strided<[4, 1], offset: ?>>
// CHECK:             [[VAR_19_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4x4xf32>
// CHECK:             bufferization.materialize_in_destination [[VAR_19_]] in writable [[VAR_reinterpret_cast_]] : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.addi [[VAR_arg15_]], [[VAR_9_]] : index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.addi [[VAR_arg16_]], [[VAR_9_]] : index
// CHECK:             scf.yield [[VAR_20_]], [[VAR_21_]] : index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
