// RUN: triton-opt --mark-gm-load %s | FileCheck %s

// ============================================================================
// Test 1 (positive): GM load inside scf.for within VECTOR scope.
// Source chain: func arg -> memref.reinterpret_cast -> copy.
// Dest: memref.alloc. Expect annotation.mark {hivm.multi_buffer = 2 : i32}.
// ============================================================================

// CHECK-LABEL: func.func @gm_load_vector_scope
func.func @gm_load_vector_scope(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
      // CHECK: memref.alloc() : memref<128xf16>
      // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<128xf16>
      %alloc = memref.alloc() : memref<128xf16>
      memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test 2 (negative): source traces to a local alloc, not a func arg.
// traceSourceToFuncArg returns false => no marking.
// ============================================================================

// CHECK-LABEL: func.func @gm_load_local_source_not_marked
func.func @gm_load_local_source_not_marked() {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      %src = memref.alloc() : memref<128xf16>
      %dst = memref.alloc() : memref<128xf16>
      memref.copy %src, %dst : memref<128xf16> to memref<128xf16>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  // CHECK-NOT: annotation.mark {{.*}}hivm.multi_buffer
  return
}

// ============================================================================
// Test 3 (negative): copy outside scf.for => not a multi-buffer candidate.
// ============================================================================

// CHECK-LABEL: func.func @gm_load_outside_loop_not_marked
func.func @gm_load_outside_loop_not_marked(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
  %alloc = memref.alloc() : memref<128xf16>
  memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
  // CHECK-NOT: annotation.mark {{.*}}hivm.multi_buffer
  return
}

// ============================================================================
// Test 4 (positive): nested scf.for — the inner-for iter_arg traces through
// the outer-for iter_arg back to a func argument.
// Source chain: func arg -> outer iter_arg -> inner iter_arg -> copy.
// ============================================================================

// CHECK-LABEL: func.func @gm_load_nested_for
func.func @gm_load_nested_for(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : index
  %c128_i32 = arith.constant 128 : index
  %c1_i32 = arith.constant 1 : index
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%carry = %arg0) -> memref<?xf16> {
      scf.for %j = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%inner_carry = %carry) -> memref<?xf16> {
        %reinterpret_cast = memref.reinterpret_cast %inner_carry to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
        // CHECK: memref.alloc() : memref<128xf16>
        // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<128xf16>
        %alloc = memref.alloc() : memref<128xf16>
        memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
        scf.yield %inner_carry : memref<?xf16>
      }
      scf.yield %carry : memref<?xf16>
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test 5 (positive): scf.while before-block arg traces back to func argument
// via whileOp.getInits().
// Source chain: func arg -> while init -> while before-block arg -> copy.
// ============================================================================

// CHECK-LABEL: func.func @gm_load_while_arg
func.func @gm_load_while_arg(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      scf.while (%while_arg = %arg0) : (memref<?xf16>) -> (memref<?xf16>) {
        %reinterpret_cast = memref.reinterpret_cast %while_arg to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
        // CHECK: memref.alloc() : memref<128xf16>
        // CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<128xf16>
        %alloc = memref.alloc() : memref<128xf16>
        memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
        scf.condition(%true) %while_arg : memref<?xf16>
      } do {
      ^bb0(%after_arg: memref<?xf16>):
        scf.yield %after_arg : memref<?xf16>
      }
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// ============================================================================
// Test 6 (positive): memref.copy in scf.while after-block.
// Source chain: func arg -> while init -> before-block arg
//               -> condition args -> after-block arg -> copy.
// ============================================================================

// CHECK-LABEL: func.func @gm_load_while_after_block
func.func @gm_load_while_after_block(%arg0: memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  scope.scope : () -> () {
    scf.for %i = %c0_i32 to %c128_i32 step %c1_i32 : i32 {
      scf.while (%before_arg = %arg0) : (memref<?xf16>) -> (memref<?xf16>) {
        scf.condition(%true) %before_arg : memref<?xf16>
      } do {
      ^bb0(%after_arg: memref<?xf16>):
        %reinterpret_cast = memref.reinterpret_cast %after_arg to offset: [%c0], sizes: [128], strides: [1] : memref<?xf16> to memref<128xf16, strided<[1], offset: ?>>
        %alloc = memref.alloc() : memref<128xf16>
        memref.copy %reinterpret_cast, %alloc : memref<128xf16, strided<[1], offset: ?>> to memref<128xf16>
        scf.yield %after_arg : memref<?xf16>
      }
    }
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}
// CHECK: memref.alloc() : memref<128xf16>
// CHECK-NEXT: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32} : memref<128xf16>
