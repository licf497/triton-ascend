// RUN: triton-opt --merge-input-init-shared-cube-block %s | FileCheck %s

module {
  // ============================================
  // Test Case 1: @test_split_init_on_cycle
  // ============================================
  // Scenario: matmulA (block 2) -> matmulB(input, init), and at the same time
  //   matmulA -> subf (block 4) -> matmulB(input). Merging matmulB's block into
  //   matmulA's block would form a cycle, so the pass falls back to splitting
  //   matmulB's init:
  //     - accumulate into a zero-filled tensor (tensor.empty + linalg.fill)
  //     - run matmul with that zero-filled accumulator
  //     - arith.addf the original init back (marked ssbuffer.add_from_matmul)
  // ============================================
  // CHECK-LABEL: func.func @test_split_init_on_cycle
  func.func @test_split_init_on_cycle(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %init_a = tensor.empty() : tensor<128x128xf32>
    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"
    %mat0 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init_a : tensor<128x128xf32>) -> tensor<128x128xf32>

    // Vector consumer of %mat0 together with the shared init forms the cycle.
    %sub = arith.subf %mat0, %b {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>

    // Consumer matmul shares input+init from %mat0.
    // CHECK: arith.constant {{.*}}ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"
    // CHECK: linalg.fill {{.*}}ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"
    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"
    // CHECK: arith.addf {{.*}}ssbuffer.add_from_matmul, ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"
    %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%mat0, %sub : tensor<128x128xf32>, tensor<128x128xf32>) outs(%mat0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mat1 : tensor<128x128xf32>
  }

  // ============================================
  // Test Case 2: @test_merge_shared_input_init
  // ============================================
  // Scenario: matmulA -> matmulB(input, init) with no extra dependency path.
  // Merging matmulB's block into matmulA's block does not create a cycle, so the
  // two matmuls end up in the same block and no split (add_from_matmul) is emitted.
  // ============================================
  // CHECK-LABEL: func.func @test_merge_shared_input_init
  func.func @test_merge_shared_input_init(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %init_a = tensor.empty() : tensor<128x128xf32>
    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"
    %mat0 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init_a : tensor<128x128xf32>) -> tensor<128x128xf32>

    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"
    // CHECK-NOT: ssbuffer.add_from_matmul
    %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "CUBE"} ins(%mat0, %b : tensor<128x128xf32>, tensor<128x128xf32>) outs(%mat0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mat1 : tensor<128x128xf32>
  }
}