# InstrumentProfileIntrinsics

## 作用

InstrumentProfileIntrinsics Pass 用于在TIR代码中插入性能分析相关的intrinsic函数调用。该Pass会在指定的代码块（通常是带有特定注解的for循环或block）前后插入 `T.start_profile_intrinsic()` 和 `T.end_profile_intrinsic()` 调用，用于标记需要进行性能分析的代码区域。这些intrinsic函数可以帮助开发者在运行时收集性能数据，例如执行时间、计算量等。

## 效果

该Pass会自动识别需要进行性能分析的代码块，并在这些代码块的前后插入配对的profile intrinsic调用。每个被插桩的代码块都会分配一个唯一的ID（例如3、5等），start和end调用使用相同的ID进行配对。这样在实际执行时，runtime可以根据这些标记收集对应代码块的性能数据。

从示例可以看出，Pass会在两个独立的for循环块（block "B" 和 block "C"）前后分别插入了profile intrinsic调用，使得这两个计算块可以被独立地进行性能分析。

## 调用该Pass前IR

```python
def main(A: T.Buffer((8, 8, 128), "int32"), B: T.Buffer((8, 8, 128), "int32"), C: T.Buffer((8, 8, 128), "int32")):
    # with T.block("root"):
    for i, j in T.grid(8, 8):
        for k, l in T.grid(8, 16):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                T.reads(A[vi, vj, vk * 16 + vl])
                T.writes(B[vi, vj, vk * 16 + vl])
                B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
        for k, l in T.grid(8, 16):
            with T.block("C"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                T.reads(B[vi, vj, vk * 16 + vl])
                T.writes(C[vi, vj, vk * 16 + vl])
                C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * 2
```


## 调用该Pass后IR

```python
def main(A: T.Buffer((8, 8, 128), "int32"), B: T.Buffer((8, 8, 128), "int32"), C: T.Buffer((8, 8, 128), "int32")):
    # with T.block("root"):
    for i, j in T.grid(8, 8):
        T.start_profile_intrinsic(3)
        for k, l in T.grid(8, 16):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                T.reads(A[vi, vj, vk * 16 + vl])
                T.writes(B[vi, vj, vk * 16 + vl])
                B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
        T.end_profile_intrinsic(3)
        T.start_profile_intrinsic(5)
        for k, l in T.grid(8, 16):
            with T.block("C"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                T.reads(B[vi, vj, vk * 16 + vl])
                T.writes(C[vi, vj, vk * 16 + vl])
                C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * 2
        T.end_profile_intrinsic(5)
```


# FP8ComputeLegalize

## 作用

FP8ComputeLegalize Pass 用于将使用FP8（8位浮点）数据类型进行计算的操作合法化（legalize）为使用更高精度浮点类型（如float16或float32）进行计算。FP8格式包括float8_e4m3fn（4位指数3位尾数）和float8_e5m2（5位指数2位尾数）两种变体。由于许多硬件不直接支持FP8计算，需要将FP8数据提升（promote）到更高精度进行计算，然后将结果转换回FP8格式进行存储。

## 效果

该Pass会扫描IR中所有使用FP8类型的计算操作，并进行以下转换：
1. **中间缓冲区提升**：将原本声明为FP8类型的中间缓冲区（如`C`）提升为指定的promote_dtype（float16或float32）
2. **输入数据转换**：在使用FP8输入数据前，插入类型提升操作，将FP8值通过位操作重新解释为高精度浮点数
3. **输出数据转换**：在写入FP8输出前，插入类型降级操作，将高精度计算结果转换回FP8格式

这些转换通过位操作（bitwise_or, bitwise_and, shift_left, shift_right等）实现，保证数据在FP8和高精度浮点格式之间正确转换，使得计算可以在支持的硬件上执行。

## 调用该Pass前IR

```python
def main(A: T.handle("float8_e5m2", "global"), B: T.handle("float8_e5m2", "global"), D: T.handle("float8_e5m2", "global")):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32})})
    A_1 = T.decl_buffer((100,), "float8_e5m2", data=A)
    B_1 = T.decl_buffer((100,), "float8_e5m2", data=B)
    D_1 = T.decl_buffer((100,), "float8_e5m2", data=D)
    C = T.decl_buffer((100,), "float8_e5m2")
    for i in range(100):
        C[i] = A_1[i] + B_1[i]
        D_1[i] = T.exp(C[i])
```


## 调用该Pass后IR

```python
def main(A: T.handle("float8_e5m2", "global"), B: T.handle("float8_e5m2", "global"), D: T.handle("float8_e5m2", "global")):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32})})
    A_1 = T.decl_buffer((100,), "float8_e5m2", data=A)
    B_1 = T.decl_buffer((100,), "float8_e5m2", data=B)
    D_1 = T.decl_buffer((100,), "float8_e5m2", data=D)
    C = T.decl_buffer((100,), "float16")  # 中间缓冲区类型从float8_e5m2提升为float16
    for i in range(100):
        # A_1[i]和B_1[i]通过位操作从float8_e5m2转换为float16后进行加法
        C[i] = T.reinterpret("float16", T.shift_left(T.Cast("uint16", T.reinterpret("uint8", A_1[i])), T.uint16(8))) + T.reinterpret("float16", T.shift_left(T.Cast("uint16", T.reinterpret("uint8", B_1[i])), T.uint16(8)))
        # T.exp(C[i])的结果通过位操作从float16转换回float8_e5m2
        D_1[i] = T.reinterpret("float8_e5m2", T.Cast("uint8", T.shift_right(T.reinterpret("uint16", T.exp(C[i])) + (T.bitwise_and(T.shift_right(T.reinterpret("uint16", T.exp(C[i])), T.uint16(8)), T.uint16(1)) + T.uint16(127)), T.uint16(8))))
```

# VerifyVTCMLimit

## 作用

VerifyVTCMLimit 是一个验证性Pass，用于检查在Hexagon架构的VTCM（Vector Tightly Coupled Memory）中分配的内存是否超过硬件限制。VTCM是Hexagon DSP上的一种高速本地内存，容量有限，该Pass确保编译时检测到的VTCM使用量不超过目标硬件的容量限制。

## 效果

该Pass遍历IR，统计所有标记为`global.vtcm`作用域的内存分配，累加其大小，并与目标设备的VTCM容量限制进行比较。如果超过限制，Pass会抛出异常终止编译，防止生成无法在目标硬件上运行的代码。

## 调用该Pass前IR

**注意：VerifyVTCMLimit是一个验证Pass，不会修改IR，仅检查VTCM内存分配是否超过硬件限制。调用前后IR保持一致。如果VTCM分配总量超过硬件限制，Pass会抛出异常终止编译。**

## 调用该Pass后IR

**注意：VerifyVTCMLimit是一个验证Pass，不会修改IR，仅检查VTCM内存分配是否超过硬件限制。调用前后IR保持一致。如果VTCM分配总量超过硬件限制，Pass会抛出异常终止编译。**

# LowerVtcmAlloc

**注意：未找到专门的测试用例文件。以下作用、效果和IR示例是基于源码分析得出。**

## 作用

LowerVtcmAlloc Pass 用于将Hexagon架构上的VTCM（Vector Tightly Coupled Memory）内存分配lowering为特定的运行时调用。该Pass将高层的`global.vtcm`作用域的allocate操作转换为TVM内置的N维内存分配函数调用。

## 效果

Pass会将IR中标记为`global.vtcm`作用域的`T.allocate`语句转换为使用`builtin::nd_mem_alloc_with_scope()`的LetStmt。具体转换包括：
1. 将allocate节点替换为LetStmt，变量绑定到`nd_mem_alloc_with_scope`调用
2. 使用`builtin::tvm_stack_make_shape()`构建shape参数
3. 传递存储作用域字符串、维度数量和shape信息

根据源码`src/tir/transforms/lower_vtcm_alloc.cc`，该Pass只处理包含"global.vtcm"的存储作用域。

## 调用该Pass前IR

N/A

## 调用该Pass后IR

N/A

# AnnotateEntryFunc

## 作用

AnnotateEntryFunc Pass 用于为IRModule中的入口函数添加`tir.is_entry_func`属性标记。在一个包含多个函数的模块中，入口函数是对外暴露的主要接口函数，需要特殊标记以便后续编译阶段进行特殊处理（如生成PackedAPI）。

## 效果

Pass会检查IRModule，如果只有一个PrimFunc，则自动将其标记为入口函数。如果有多个函数且没有明确指定哪个是入口函数，Pass可能会失败或需要额外的配置。

## 调用该Pass前IR

```python
@T.prim_func(private=True)
def func1(A: T.Buffer((16,), "float32")):
    for i in range(16):
        if i == 5:
            if i == 5:
                A[i] = T.float32(0.0)
```

## 调用该Pass后IR

```python
@T.prim_func(private=True)
def func1(A: T.Buffer((16,), "float32")):
    T.func_attr({"tir.is_entry_func": T.bool(True)})  # 添加了入口函数标记
    for i in range(16):
        if i == 5:
            if i == 5:
                A[i] = T.float32(0.0)
```

# ThreadSync

## 作用

ThreadSync Pass 用于在多线程并行代码中插入必要的同步原语（synchronization primitives），确保共享内存访问的正确性。该Pass主要针对GPU编程，在对共享内存（shared memory）或其他需要同步的存储区域进行读写时，插入适当的内存屏障（barrier）指令。

## 效果

Pass分析IR中的数据依赖关系，识别出需要同步的位置。当检测到一个线程写入共享内存后，另一个线程读取同一位置时，会在写入和读取之间插入`T.tvm_storage_sync()`调用，确保所有线程的写操作对其他线程可见。

## 调用该Pass前IR

```python
@T.prim_func(private=True)
def main(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
    blockIdx_x = T.launch_thread("blockIdx.x", 1)
    B_1 = T.allocate([24], "float32", "shared.dyn")
    C_1 = T.allocate([1], "float32", "local")
    D_1 = T.allocate([16], "float32", "shared.dyn")
    threadIdx_x = T.launch_thread("threadIdx.x", 16)
    B_1_1 = T.Buffer((24,), data=B_1, scope="shared.dyn")
    A_1 = T.Buffer((16,), data=A.data)
    B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
    C_1_1 = T.Buffer((1,), data=C_1, scope="local")
    C_1_1[0] = B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
    D_1_1 = T.Buffer((16,), data=D_1, scope="shared.dyn")
    D_1_1[threadIdx_x] = C_1_1[0]  # 写入shared memory
    E_1 = T.Buffer((16,), data=E.data)
    E_1[threadIdx_x] = D_1_1[threadIdx_x]  # 读取shared memory，需要同步
```

## 调用该Pass后IR

```python
@T.prim_func(private=True)
def main(A: T.Buffer((4, 4), "float32"), E: T.Buffer((4, 4), "float32")):
    blockIdx_x = T.launch_thread("blockIdx.x", 1)
    B_1 = T.allocate([24], "float32", "shared.dyn")
    C_1 = T.allocate([1], "float32", "local")
    D_1 = T.allocate([16], "float32", "shared.dyn")
    threadIdx_x = T.launch_thread("threadIdx.x", 16)
    B_1_1 = T.Buffer((24,), data=B_1, scope="shared.dyn")
    A_1 = T.Buffer((16,), data=A.data)
    B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
    C_1_1 = T.Buffer((1,), data=C_1, scope="local")
    C_1_1[0] = B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
    T.tvm_storage_sync("shared.dyn")  # 插入同步调用
    D_1_1 = T.Buffer((16,), data=D_1, scope="shared.dyn")
    D_1_1[threadIdx_x] = C_1_1[0]
    E_1 = T.Buffer((16,), data=E.data)
    E_1[threadIdx_x] = D_1_1[threadIdx_x]
```

# InferFragment

**注意：未找到专门的测试用例文件。以下作用、效果和IR示例是基于源码分析得出。**

## 作用

InferFragment Pass 用于从TensorCore相关的intrinsic函数调用中推断fragment（片段）元数据信息。该Pass分析`tvm_load_matrix_sync`、`tvm_store_matrix_sync`、`tvm_fill_fragment`和`tvm_mma_sync`等TensorCore操作，提取矩阵维度（m, n, k）和布局信息，并将这些元数据作为属性附加到相应的allocate节点上。

## 效果

Pass会遍历IR，从TensorCore intrinsic调用中收集fragment信息，包括：
1. **矩阵维度**：提取m、n、k维度信息
2. **布局信息**：对于`wmma.matrix_a`和`wmma.matrix_b`作用域的buffer，提取布局（row_major或col_major）
3. **属性添加**：为fragment buffer的allocate节点添加`fragment_shape`属性（包含"m, n, k"字符串），对于matrix_a和matrix_b还添加`fragment_layout`属性

根据源码`src/tir/transforms/tensorcore_infer_fragment.cc`，该Pass确保所有使用的fragment在使用前都已被正确初始化（通过fill或load操作）。

## 调用该Pass前IR

N/A

## 调用该Pass后IR

N/A

# LowerThreadAllreduce

## 作用

LowerThreadAllreduce Pass 用于将线程级的归约操作（thread-level all-reduce）lowering为具体的实现。该Pass将高层的`T.tvm_thread_allreduce`调用转换为一系列的warp shuffle指令或shared memory操作，实现线程间的高效数据聚合。

## 效果

Pass会分析归约操作的类型（sum, max, min等）、数据类型和线程配置，选择最优的实现策略。对于小规模归约（如单个warp内），使用warp shuffle指令；对于大规模归约，使用shared memory和多级归约。

从示例可以看出，对于32个线程的归约操作，Pass使用了一系列`tvm_warp_shuffle_down`指令进行树形归约：先偏移16、8、4、2、1，最后使用`tvm_warp_shuffle`将结果广播到所有线程。

## 调用该Pass前IR

```python
@T.prim_func(private=True)
def main(A: T.Buffer((128, 32), "float32"), B: T.Buffer((128,), "float32")):
    T.func_attr({"target": T.target({"arch": "sm_50", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    for i in range(128):
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        reduce = T.allocate([1], "float32", "local")
        reduce_1 = T.Buffer((1,), data=reduce, scope="local")
        with T.attr(T.comm_reducer(lambda x, y: x + y, [T.float32(0.0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0))):
            A_flat = T.Buffer((4096,), data=A.data)
            T.tvm_thread_allreduce(T.uint32(1), A_flat[0], T.bool(True), reduce_1[0], threadIdx_x)
        if threadIdx_x == 0:
            B[i] = reduce_1[0]
```

## 调用该Pass后IR

```python
@T.prim_func(private=True)
def main(A: T.Buffer((128, 32), "float32"), B: T.Buffer((128,), "float32")):
    T.func_attr({"target": T.target({"arch": "sm_50", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    for i in range(128):
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        red_buf0 = T.allocate([1], "float32", "local")
        red_buf0_1 = T.Buffer((1,), data=red_buf0, scope="local")
        with T.attr(T.comm_reducer(lambda x, y: x + y, [T.float32(0.0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0))):
            mask = T.decl_buffer((1,), "uint32", scope="local")
            t0 = T.decl_buffer((1,), scope="local")
            A_flat = T.Buffer((4096,), data=A.data)
            red_buf0_1[0] = A_flat[0]
            mask[0] = T.tvm_warp_activemask()
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            t0[0] = T.tvm_warp_shuffle_down(mask[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0[0]
            red_buf0_1[0] = T.tvm_warp_shuffle(mask[0], red_buf0_1[0], 0, 32, 32)
        if threadIdx_x == 0:
            B[i] = red_buf0_1[0]
```

# InjectPTXAsyncCopy

## 作用

InjectPTXAsyncCopy Pass 用于将符合特定模式的内存拷贝操作转换为NVIDIA PTX异步拷贝指令（cp.async）。该Pass识别在`async_scope`属性标记下的从global memory到shared memory的内存拷贝操作，并将其替换为硬件加速的异步拷贝指令。异步拷贝可以在数据传输的同时执行计算，提高GPU利用率。该Pass仅适用于支持异步拷贝的NVIDIA GPU架构（Ampere及更新）。

## 效果

Pass扫描IR中标记为`async_scope`的代码块，识别从global memory到shared memory的赋值操作，并将其转换为`T.ptx_cp_async`调用。转换后：
1. **移除async_scope属性**：原来的`T.attr("default", "async_scope", 1)`属性块被移除
2. **替换内存拷贝**：将`A_shared[...] = A[...]`形式的赋值替换为`T.ptx_cp_async(...)`调用
3. **保留同步操作**：`T.ptx_commit_group()`和`T.ptx_wait_group(0)`调用被保留，确保数据在使用前传输完成

从示例可以看出，对于128次循环的内存拷贝，每次拷贝4字节的float32数据，Pass将标准的赋值操作转换为了ptx_cp_async调用，参数包括数据类型、目标地址、目标偏移、源地址、源偏移和字节数。

## 调用该Pass前IR

```python
@T.prim_func
def main(A: T.Buffer((32, 128), "float32"), B: T.Buffer((32, 128), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    bx = T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    A_shared = T.allocate([4096], "float32", "shared")
    T.attr("default", "async_scope", 1)
    A_shared_1 = T.Buffer((4096,), data=A_shared, scope="shared")
    for i in range(128):
        A_1 = T.Buffer((4096,), data=A.data)
        A_shared_1[tx * 128 + i] = A_1[tx * 128 + i]
    T.ptx_commit_group()
    T.ptx_wait_group(0)
    for i in range(128):
        B_1 = T.Buffer((4096,), data=B.data)
        B_1[tx * 128 + i] = A_shared_1[tx * 128 + i]
```

## 调用该Pass后IR

```python
@T.prim_func
def main(A: T.Buffer((32, 128), "float32"), B: T.Buffer((32, 128), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    bx = T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    A_shared = T.allocate([4096], "float32", "shared")
    for i in range(128):
        T.ptx_cp_async("float32", A_shared, tx * 128 + i, A.data, tx * 128 + i, 4)
    T.ptx_commit_group()
    T.ptx_wait_group(0)
    for i in range(128):
        B_1 = T.Buffer((4096,), data=B.data)
        A_shared_1 = T.Buffer((4096,), data=A_shared, scope="shared")
        B_1[tx * 128 + i] = A_shared_1[tx * 128 + i]
```

# InjectPTXLDG32

## 作用

InjectPTXLDG32 Pass 用于将条件内存读取操作（使用`T.if_then_else`的全局内存加载）转换为NVIDIA PTX的`ldg.32`指令。该Pass专门处理包含条件判断的全局内存读取模式，将其优化为使用硬件支持的predicated load指令。`ptx_ldg32`是一个带谓词（predicated）的全局内存加载intrinsic，可以在条件为真时从全局内存加载数据，为假时使用默认值，避免分支。该Pass仅适用于NVIDIA GPU架构（要求SM80及以上）。

## 效果

Pass识别包含`T.if_then_else`模式的BufferStore操作，其中条件的真值分支是全局内存读取，假值分支是立即数。Pass会：
1. **创建辅助buffer**：分配`addr`和`predicate` buffer用于存储地址和条件
2. **展开条件表达式**：将`T.if_then_else(predicate, A[global_idx], 0.0)`分解为：
   - 存储全局地址到`addr[0]`
   - 存储本地地址到`addr[1]`
   - 存储predicate到`predicate[0]`
   - 先写入默认值到local buffer
   - 调用`T.ptx_ldg32`在predicate为真时覆盖写入实际值

从示例可以看出，对于`A_local[tx] = T.if_then_else(tx % 2 == 0, A[tx // 2], 0.0)`这样的条件加载，Pass将其转换为一系列操作，最后通过`T.ptx_ldg32(A_local, predicate, A[addr], local_addr)`完成predicated load，避免了warp内的分支divergence。

## 调用该Pass前IR

```python
@T.prim_func
def default_function(A: T.Buffer((16,), "float32"), B: T.Buffer((32,), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    bx = T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    A_local = T.allocate([32], "float32", "local")
    A_local_1 = T.Buffer((32,), data=A_local, scope="local")
    A_1 = T.Buffer((16,), data=A.data)
    A_local_1[tx] = T.if_then_else(tx % 2 == 0, A_1[tx // 2], T.float32(0.0))
    B_1 = T.Buffer((32,), data=B.data)
    B_1[tx] = A_local_1[tx] + T.float32(1.0)
```

## 调用该Pass后IR

```python
@T.prim_func
def default_function(A: T.Buffer((16,), "float32"), B: T.Buffer((32,), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    bx = T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    predicate = T.allocate([1], "bool", "local")
    addr = T.allocate([2], "int32", "local")
    A_local = T.allocate([32], "float32", "local")
    addr_1 = T.Buffer((2,), "int32", data=addr, scope="local")
    addr_1[0] = tx // 2
    addr_1[1] = tx
    predicate_1 = T.Buffer((1,), "bool", data=predicate, scope="local")
    predicate_1[0] = tx % 2 == 0
    A_local_1 = T.Buffer((32,), data=A_local, scope="local")
    A_local_1[addr_1[1]] = T.float32(0.0)
    A_1 = T.Buffer((16,), data=A.data)
    T.ptx_ldg32(A_local, predicate_1[0], A_1[addr_1[0]], addr_1[1])
    B_1 = T.Buffer((32,), data=B.data)
    B_1[tx] = A_local_1[tx] + T.float32(1.0)
```

# AnnotateDeviceRegions

## 作用

AnnotateDeviceRegions Pass 用于在异构编程（host-device分离）场景中标注IR中哪些代码区域应该在设备（如GPU）上执行。该Pass是SplitHostDevice Pass的前置步骤，它识别包含设备特定属性的代码区域并添加目标设备注解。Pass只在函数具有host target（即`target.GetHost()`不为空）时生效，将函数体内需要在设备上执行的代码区域用`T.attr(device_target, "target", 0)`包裹。
AnnotateDeviceRegions是一个通用的异构编程Pass，只要是host-device分离的场景就可以使用，不限定特定的GPU或硬件平台。

## 效果

Pass遍历IR，识别以下设备特定的属性节点：
1. **`attr::thread_extent`**：线程绑定（如`threadIdx.x`, `blockIdx.x`）
2. **`attr::pipeline_exec_scope`**：流水线执行作用域
3. **`attr::device_scope`**：显式的设备作用域标记

当遇到这些属性时，Pass会在其外层添加一个`T.attr(device_target, "target", 0)`属性节点，其中`device_target`是函数target去除host部分后的设备target（通过`target.WithoutHost()`获得）。这个注解告诉后续的SplitHostDevice Pass，被包裹的代码应该被提取为独立的设备kernel函数。

如果代码中已经存在`tvm::attr::kTarget`属性，Pass会保持原样不修改。

从示例可以看出，对于包含`T.launch_thread("threadIdx.x", 16)`的代码，Pass在其外层添加了`T.attr(T.target("cuda"), "target", 0)`注解，标记这段代码需要在CUDA设备上执行。

## 调用该Pass前IR

```python
@T.prim_func
def before(A: T.Buffer((16,), "float32")):
    T.func_attr({"target": T.target({"arch": "sm_50", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    i = T.launch_thread("threadIdx.x", 16)
    A[i] = T.float32(0.0)
```

## 调用该Pass后IR

```python
@T.prim_func
def before(A: T.Buffer((16,), "float32")):
    T.func_attr({"target": T.target({"arch": "sm_50", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    T.attr(T.target({"arch": "sm_50", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "target", 0)
    i = T.launch_thread("threadIdx.x", 16)
    A[i] = T.float32(0.0)
```
        # GPU计算
        ...
```

# SplitHostDevice

## 作用

SplitHostDevice Pass 用于将包含主机（host）和设备（device）混合代码的函数分离为独立的主机函数和设备kernel函数。该Pass是异构编程中的关键步骤，通过识别`T.attr(target, "target", 0)`标记（由AnnotateDeviceRegions Pass添加），将设备代码提取为独立的kernel函数，使得主机代码和设备代码可以分别编译和优化。

Pass会分析设备代码块中使用的变量（通过VarUseDefAnalyzer），提取未定义变量作为kernel函数的参数，确保所有kernel需要的数据都通过参数传递。对于某些设备类型（如CPU、ExtDev、Hexagon），生成的kernel可以返回int32错误码，主机端会插入错误检查代码。

SplitHostDevice是一个通用的异构编程Pass，适用于所有host-device分离场景，不限定特定硬件平台（CUDA、ROCm、Metal、Vulkan、OpenCL、WebGPU、Hexagon等均可使用）。

## 效果

Pass执行以下转换：
1. **识别设备代码区域**：查找包含`attr::kTarget`属性的代码块（由AnnotateDeviceRegions添加）
2. **参数提取**：使用VarUseDefAnalyzer分析设备代码中的未定义变量（即需要从host传入的参数），按类型和名称排序
3. **创建设备kernel**：
   - 提取设备代码，创建新的PrimFunc，函数名为`原函数名_kernel`
   - 设置kernel的target为纯设备target（去除host部分）
   - 添加`tir.noalias`和`tir.is_global_func`属性
   - 对于支持错误传播的设备（CPU/ExtDev/Hexagon），kernel返回int32状态码
4. **修改主机函数**：将设备代码区域替换为对kernel函数的调用
5. **错误处理**：对于可返回错误码的kernel，主机端插入`assert kernel_error_code == 0`检查
6. **SSA转换**：最后对整个模块执行ConvertSSA，确保host和device函数使用独立的TIR变量

从示例可以看出，Pass将包含`T.attr(cuda_target, "target", 0)`的设备代码提取为`main_kernel`函数，原`main`函数变为纯主机函数，通过`Module.main_kernel(n)`调用设备kernel。

## 调用该Pass前IR

```python
@T.prim_func
def main(n: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_50", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "opt-level": 0, "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    T.attr(T.target({"arch": "sm_50", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "target", 0)
    T.evaluate(n)
```

## 调用该Pass后IR

```python
@T.prim_func(private=True)
def main_kernel(n: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_50", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
    T.evaluate(n)

@T.prim_func
def main(n: T.int32):
    T.func_attr({"target": T.target({"arch": "sm_50", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "opt-level": 0, "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    Module.main_kernel(n)
```

# MergeSharedMemoryAllocations

## 作用

MergeSharedMemoryAllocations Pass 用于合并GPU kernel中的多个shared memory分配，减少shared memory的总使用量。该Pass通过分析shared memory buffer的生命周期（liveness analysis），将不同时使用的buffer合并到同一块内存空间中，从而提高shared memory利用率，允许更多的线程块并发执行。

Pass处理两种类型的shared memory分配：
1. **动态shared memory**（`shared.dyn`）：每个GPU kernel默认只允许一个动态shared memory分配
2. **静态shared memory**（`shared`）：可通过`tir.merge_static_smem`配置选项控制是否合并

该Pass适用于支持shared memory的GPU平台，包括NVIDIA CUDA、AMD ROCm、Vulkan（SPIRV）、WebGPU等。

## 效果

Pass执行以下转换：
1. **收集分配**：扫描IR，收集所有`shared.dyn`和`shared`作用域的allocate节点
2. **生命周期分析**：通过`SharedMemLinearAccessPatternFinder`分析每个buffer的使用范围，构建线性访问模式
3. **内存复用规划**：对生命周期不重叠的buffer进行复用规划，计算最优的内存布局和偏移
4. **合并分配**：
   - 创建单个`buf_dyn_shmem`分配，类型为`uint8`，大小为所有buffer所需字节数之和
   - 将原来的多个独立allocate替换为对`buf_dyn_shmem`的引用，使用不同的偏移
   - 更新所有buffer访问的索引，加上对应的偏移量
5. **保留DeclBuffer**：如果原来有`T.decl_buffer`声明，合并后保留该声明

从示例可以看出，Pass将两个独立的128元素float32 buffer（`A_sh`和`B_sh`，各512字节）合并为一个512字节的`buf_dyn_shmem`。由于两个buffer的生命周期重叠（都在函数体内使用），它们被分配到同一块内存的不同位置（A_sh复用同一起始地址，B_sh不需要偏移因为可以复用）。

## 调用该Pass前IR

```python
@T.prim_func
def func():
    threadIdx_x = T.launch_thread("threadIdx.x", 128)
    A_sh = T.allocate([128], "float32", "shared.dyn")
    B_sh = T.allocate([128], "float32", "shared.dyn")
    A_sh_1 = T.decl_buffer((128,), data=A_sh, scope="shared.dyn")
    B_sh_1 = T.decl_buffer((128,), data=B_sh, scope="shared.dyn")
    A_sh_1[threadIdx_x] = T.float32(0.0)
    B_sh_1[threadIdx_x] = T.float32(0.0)
```

## 调用该Pass后IR

```python
@T.prim_func
def func():
    threadIdx_x = T.launch_thread("threadIdx.x", 128)
    buf_dyn_shmem = T.allocate([512], "uint8", "shared.dyn")
    A_sh = T.decl_buffer((128,), data=buf_dyn_shmem, scope="shared.dyn")
    B_sh = T.decl_buffer((128,), data=buf_dyn_shmem, scope="shared.dyn")
    A_sh[threadIdx_x] = T.float32(0.0)
    B_sh[threadIdx_x] = T.float32(0.0)
```

# MakePackedAPI

## 作用

MakePackedAPI Pass 用于将PrimFunc转换为PackedFunc接口，使其可以被TVM runtime调用。PackedFunc是TVM中统一的函数调用接口，支持动态类型和参数打包，便于跨语言调用和动态分发。

该Pass只处理具有`global_symbol`属性的函数（对外暴露的公共函数），内部子函数不需要转换。Pass要求函数必须有`target`属性，并且该target包含host部分（即异构编程场景）。转换后，函数签名从类型化参数变为统一的PackedFunc签名。

MakePackedAPI是一个通用Pass，适用于所有需要与TVM runtime交互的场景，不限定特定硬件平台。

## 效果

Pass执行以下转换：
1. **修改函数签名**：将原来的Buffer参数转换为PackedFunc的标准签名：
   - `args: T.handle` - 参数数组指针
   - `arg_type_ids: T.handle("int32")` - 参数类型码数组
   - `num_args: T.int32` - 参数数量
   - `out_ret_value: T.handle("void")` - 返回值指针
   - `out_ret_tcode: T.handle("int32")` - 返回值类型码
   - `resource_handle: T.handle` - 资源句柄
   - 返回类型变为`T.int32`（错误码）

2. **参数校验**：添加大量断言检查参数有效性：
   - `num_args`数量检查
   - 空指针检查（`T.isnullptr`）
   - 参数类型码验证（pointer类型检查）
   - Buffer元数据验证（ndim、dtype、shape、strides、byte_offset、device_type等）

3. **参数解包**：从packed格式提取实际参数：
   - 使用`T.tvm_struct_get`从args数组获取DLTensor指针
   - 从DLTensor结构体提取shape、strides、data指针等
   - 使用`T.decl_buffer`创建Buffer视图

4. **设置执行环境**：
   - 添加`T.attr("default", "device_id", dev_id)`设置设备ID
   - 添加`T.attr("default", "device_type", 1)`设置设备类型
   - 为buffer添加`T.attr(buffer, "storage_alignment", 64)`对齐属性

5. **计算作用域包裹**：将原函数体包裹在`T.attr(0, "compute_scope", "函数名_compute_")`中

6. **调用约定和Target更新**：
   - 添加`"calling_conv": 1`属性（PackedFunc调用约定）
   - 移除target中的host部分，只保留host target

从示例可以看出，简单的两个Buffer参数函数被转换为包含6个标准PackedFunc参数的函数，添加了详尽的参数校验和解包逻辑，原计算代码被包裹在`compute_scope`中。

## 调用该Pass前IR

```python
@T.prim_func
def main(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
    T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""}, "keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    for i in range(16):
        B[i] = A[i] * T.float32(2.0)
```

## 调用该Pass后IR

```python
@T.prim_func
def main(args: T.handle, arg_type_ids: T.handle("int32"), num_args: T.int32, out_ret_value: T.handle("void"), out_ret_tcode: T.handle("int32"), resource_handle: T.handle) -> T.int32:
    T.func_attr({"calling_conv": 1, "target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    # 参数数量检查
    assert num_args == 2, "main: num_args should be 2"
    assert not T.isnullptr(args), "main: TVMValue* arg pointer was NULL"
    assert not T.isnullptr(arg_type_ids), "main: int* type_codes was NULL"
    
    # 参数类型码检查
    arg_type_ids_1 = T.decl_buffer((2,), "int32", data=arg_type_ids)
    A_handle_code: T.int32 = arg_type_ids_1[0]
    assert A_handle_code == 3 or A_handle_code == 13 or A_handle_code == 7 or A_handle_code == 4, "main: Expect arg[0] to be pointer"
    B_handle_code: T.int32 = arg_type_ids_1[1]
    assert B_handle_code == 3 or B_handle_code == 13 or B_handle_code == 7 or B_handle_code == 4, "main: Expect arg[1] to be pointer"
    
    # 从packed args提取DLTensor指针
    A_handle: T.handle = T.tvm_struct_get(args, 0, 12, "handle")
    B_handle: T.handle = T.tvm_struct_get(args, 1, 12, "handle")
    
    # A buffer的元数据验证
    assert not T.isnullptr(A_handle), "main.A_handle is expected to have non-NULL DLTensor* pointer"
    assert 1 == T.tvm_struct_get(A_handle, 0, 4, "int32"), "main.A_handle.ndim is expected to equal 1"
    main_A_handle_shape: T.handle("int64") = T.tvm_struct_get(A_handle, 0, 2, "handle")
    main_A_handle_shape_1 = T.decl_buffer((1,), "int64", data=main_A_handle_shape)
    main_A_handle_strides: T.handle("int64") = T.tvm_struct_get(A_handle, 0, 3, "handle")
    main_A_handle_strides_1 = T.decl_buffer((0,), "int64", data=main_A_handle_strides)
    dev_id: T.int32 = T.tvm_struct_get(A_handle, 0, 9, "int32")
    A: T.handle("float32", "global") = T.tvm_struct_get(A_handle, 0, 1, "handle")
    T.attr(A, "storage_alignment", 64)
    
    # B buffer的元数据验证（类似A）
    assert not T.isnullptr(B_handle), "main.B_handle is expected to have non-NULL DLTensor* pointer"
    assert 1 == T.tvm_struct_get(B_handle, 0, 4, "int32"), "main.B_handle.ndim is expected to equal 1"
    main_B_handle_shape: T.handle("int64") = T.tvm_struct_get(B_handle, 0, 2, "handle")
    main_B_handle_shape_1 = T.decl_buffer((1,), "int64", data=main_B_handle_shape)
    main_B_handle_strides: T.handle("int64") = T.tvm_struct_get(B_handle, 0, 3, "handle")
    main_B_handle_strides_1 = T.decl_buffer((0,), "int64", data=main_B_handle_strides)
    B: T.handle("float32", "global") = T.tvm_struct_get(B_handle, 0, 1, "handle")
    T.attr(B, "storage_alignment", 64)
    
    # 设置设备环境
    T.attr("default", "device_id", dev_id)
    T.attr("default", "device_type", 1)
    
    # dtype、shape、strides、offset验证
    assert T.tvm_struct_get(A_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(A_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(A_handle, 0, 7, "uint16") == T.uint16(1), "main.A_handle.dtype is expected to be float32"
    assert T.Cast("int32", main_A_handle_shape_1[0]) == 16, "Argument main.A_handle.shape[0] has an unsatisfied constraint: 16 == T.Cast(\"int32\", main_A_handle_shape[0])"
    if not T.isnullptr(main_A_handle_strides):
        assert 1 == T.Cast("int32", main_A_handle_strides_1[0]), "main.A_handle.strides: expected to be compact array"
        T.evaluate(0)
    assert T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, "uint64"), "Argument main.A_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, \"uint64\")"
    assert T.tvm_struct_get(A_handle, 0, 10, "int32") == 1, "Argument main.A_handle.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(A_handle, 0, 10, \"int32\")"
    assert not T.isnullptr(A), "main.A_handle is expected to have non-NULL data pointer"
    
    # B buffer验证（类似A）
    assert T.tvm_struct_get(B_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(B_handle, 0, 6, "uint8") == T.uint8(32) and T.tvm_struct_get(B_handle, 0, 7, "uint16") == T.uint16(1), "main.B_handle.dtype is expected to be float32"
    assert T.Cast("int32", main_B_handle_shape_1[0]) == 16, "Argument main.B_handle.shape[0] has an unsatisfied constraint: 16 == T.Cast(\"int32\", main_B_handle_shape[0])"
    if not T.isnullptr(main_B_handle_strides):
        assert 1 == T.Cast("int32", main_B_handle_strides_1[0]), "main.B_handle.strides: expected to be compact array"
        T.evaluate(0)
    assert T.uint64(0) == T.tvm_struct_get(B_handle, 0, 8, "uint64"), "Argument main.B_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(B_handle, 0, 8, \"uint64\")"
    assert T.tvm_struct_get(B_handle, 0, 10, "int32") == 1, "Argument main.B_handle.device_type has an unsatisfied constraint: 1 == T.tvm_struct_get(B_handle, 0, 10, \"int32\")"
    assert dev_id == T.tvm_struct_get(B_handle, 0, 9, "int32"), "Argument main.B_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(B_handle, 0, 9, \"int32\")"
    assert not T.isnullptr(B), "main.B_handle is expected to have non-NULL data pointer"
    
    # 创建buffer视图
    A_1 = T.decl_buffer((16,), data=A)
    B_1 = T.decl_buffer((16,), data=B)
    
    # 计算作用域
    with T.attr(0, "compute_scope", "main_compute_"):
        for i in range(16):
            B_1[i] = A_1[i] * T.float32(2.0)
    return 0
```

# FP8StorageLegalize

## 作用

FP8StorageLegalize Pass 用于将FP8类型的存储操作合法化。与FP8ComputeLegalize处理计算不同，该Pass专注于处理FP8数据的存储和加载，将其转换为硬件支持的格式（uint8）。该Pass在FP8ComputeLegalize之后运行，处理已经完成计算合法化的IR。

Pass检查目标硬件是否原生支持FP8（通过`tvm.contrib.nvcc.supports_fp8`），如果支持则跳过转换。否则，将所有FP8类型的buffer声明和访问转换为uint8类型，并移除不必要的`reinterpret`操作。

FP8StorageLegalize主要针对NVIDIA GPU（检查CUDA compute capability），但原理上可扩展到其他需要FP8存储合法化的平台。

## 效果

Pass执行以下转换：
1. **Buffer类型转换**：将handle参数和buffer声明中的FP8类型（`float8_e4m3fn`或`float8_e5m2`）替换为`uint8`
2. **移除冗余reinterpret**：
   - 将`T.reinterpret("uint8", T.reinterpret("float8_xxx", value))`简化为直接访问
   - 将`T.reinterpret("float8_xxx", T.reinterpret("uint8", value))`简化为直接访问
   - 保留必要的reinterpret操作（如从uint8到float16/float32的转换）
3. **变量重映射**：更新所有涉及FP8 buffer的变量声明，将其指针类型从`handle("float8_xxx")`改为`handle("uint8")`

转换后，FP8数据以uint8形式存储在内存中，与计算时的类型提升操作（FP8ComputeLegalize添加的）配合，实现完整的FP8数据流处理。

从示例可以看出，Pass将所有`handle("float8_e5m2")`转换为`handle("uint8")`，将buffer声明从`"float8_e5m2"`改为`"uint8"`，并移除了`T.reinterpret("uint8", A_1[i])`和`T.reinterpret("float8_e5m2", ...)`中的冗余转换。

## 调用该Pass前IR

```python
@T.prim_func
def main(A: T.handle("float8_e5m2", "global"), B: T.handle("float8_e5m2", "global"), D: T.handle("float8_e5m2", "global")):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32})})
    A_1 = T.decl_buffer((100,), "float8_e5m2", data=A)
    B_1 = T.decl_buffer((100,), "float8_e5m2", data=B)
    D_1 = T.decl_buffer((100,), "float8_e5m2", data=D)
    C = T.decl_buffer((100,), "float16")
    for i in range(100):
        # 读取FP8数据时需要reinterpret到uint8
        C[i] = T.reinterpret("float16", T.shift_left(T.Cast("uint16", T.reinterpret("uint8", A_1[i])), T.uint16(8))) + T.reinterpret("float16", T.shift_left(T.Cast("uint16", T.reinterpret("uint8", B_1[i])), T.uint16(8)))
        # 写入FP8数据时需要reinterpret回float8_e5m2
        D_1[i] = T.reinterpret("float8_e5m2", T.Cast("uint8", T.shift_right(T.reinterpret("uint16", T.exp(C[i])) + (T.bitwise_and(T.shift_right(T.reinterpret("uint16", T.exp(C[i])), T.uint16(8)), T.uint16(1)) + T.uint16(127)), T.uint16(8))))
```

## 调用该Pass后IR

```python
@T.prim_func
def main(A: T.handle("uint8", "global"), B: T.handle("uint8", "global"), D: T.handle("uint8", "global")):
    T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "l2_cache_size_bytes": 41943040, "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32})})
    A_1 = T.decl_buffer((100,), "uint8", data=A)
    B_1 = T.decl_buffer((100,), "uint8", data=B)
    D_1 = T.decl_buffer((100,), "uint8", data=D)
    C = T.decl_buffer((100,), "float16")
    for i in range(100):
        # 移除了T.reinterpret("uint8", ...)，直接访问uint8 buffer
        C[i] = T.reinterpret("float16", T.shift_left(T.Cast("uint16", A_1[i]), T.uint16(8))) + T.reinterpret("float16", T.shift_left(T.Cast("uint16", B_1[i]), T.uint16(8)))
        # 移除了T.reinterpret("float8_e5m2", ...)，直接写入uint8
        D_1[i] = T.Cast("uint8", T.shift_right(T.reinterpret("uint16", T.exp(C[i])) + (T.bitwise_and(T.shift_right(T.reinterpret("uint16", T.exp(C[i])), T.uint16(8)), T.uint16(1)) + T.uint16(127)), T.uint16(8)))
```

# BF16StorageLegalize

## 作用

BF16StorageLegalize Pass 用于将BF16（Brain Float 16）类型的存储操作合法化。与BF16ComputeLegalize处理计算不同，该Pass专注于处理BF16数据的存储和加载，将其转换为硬件支持的格式（uint16）。该Pass在BF16ComputeLegalize之后运行，处理已经完成计算合法化的IR。

Pass检查目标硬件是否原生支持BF16（通过`tvm.contrib.nvcc.supports_bf16`），如果支持则跳过转换。否则，将所有BF16类型的buffer声明和访问转换为uint16类型，并移除不必要的`reinterpret`操作。

BF16StorageLegalize主要针对NVIDIA GPU（检查CUDA compute capability），对于不支持原生BF16的GPU（如RTX 2080 Ti/sm_75），会进行存储合法化；对于支持原生BF16的GPU（如RTX 3090 Ti/sm_86+），则保持BF16类型不变。

## 效果

Pass执行以下转换：
1. **Buffer类型转换**：将handle参数和buffer声明中的`bfloat16`类型替换为`uint16`
2. **移除冗余reinterpret**：
   - 将`T.reinterpret("uint16", T.reinterpret("bfloat16", value))`简化为直接访问
   - 将`T.reinterpret("bfloat16", T.reinterpret("uint16", value))`简化为直接访问
   - 保留必要的reinterpret操作（如从uint16到float32的转换）
3. **变量重映射**：更新所有涉及BF16 buffer的变量声明，将其指针类型从`handle("bfloat16")`改为`handle("uint16")`
4. **保留存储作用域**：转换过程保留buffer的storage_scope属性（如"shared"、"local"）

转换后，BF16数据以uint16形式存储在内存中，与计算时的类型提升操作（BF16ComputeLegalize添加的）配合，实现完整的BF16数据流处理。

从示例可以看出，Pass将所有`handle("bfloat16")`转换为`handle("uint16")`，将buffer声明从`"bfloat16"`改为`"uint16"`，并移除了`T.reinterpret("uint16", A_1[i])`和`T.reinterpret("bfloat16", ...)`中的冗余转换，同时保留了"shared"和"local"等存储作用域。

## 调用该Pass前IR

以下IR是经过BF16ComputeLegalize处理后的结果（BF16StorageLegalize的输入）：

```python
@T.prim_func
def main(A: T.handle("bfloat16", "shared"), B: T.handle("bfloat16", "local"), D: T.handle("bfloat16", "global")):
    T.func_attr({"target": T.target({"arch": "sm_75", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32})})
    A_1 = T.decl_buffer((100,), "bfloat16", data=A, scope="shared")
    B_1 = T.decl_buffer((100,), "bfloat16", data=B, scope="local")
    D_1 = T.decl_buffer((100,), "bfloat16", data=D)
    C = T.decl_buffer((100,))
    for i in range(100):
        # 读取BF16数据时需要reinterpret到uint16再转换到float32
        C[i] = T.reinterpret("float32", T.shift_left(T.Cast("uint32", T.reinterpret("uint16", A_1[i])), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.Cast("uint32", T.reinterpret("uint16", B_1[i])), T.uint32(16)))
        # 写入BF16数据时需要reinterpret回bfloat16
        D_1[i] = T.reinterpret("bfloat16", T.Cast("uint16", T.shift_right(T.reinterpret("uint32", T.exp(C[i])) + (T.bitwise_and(T.shift_right(T.reinterpret("uint32", T.exp(C[i])), T.uint32(16)), T.uint32(1)) + T.uint32(32767)), T.uint32(16))))
```

## 调用该Pass后IR

```python
@T.prim_func
def main(A: T.handle("uint16", "shared"), B: T.handle("uint16", "local"), D: T.handle("uint16", "global")):
    T.func_attr({"target": T.target({"arch": "sm_75", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "registers_per_block": 65536, "tag": "", "thread_warp_size": 32})})
    A_1 = T.decl_buffer((100,), "uint16", data=A, scope="shared")
    B_1 = T.decl_buffer((100,), "uint16", data=B, scope="local")
    D_1 = T.decl_buffer((100,), "uint16", data=D)
    C = T.decl_buffer((100,))
    for i in range(100):
        # 移除了T.reinterpret("uint16", ...)，直接访问uint16 buffer
        C[i] = T.reinterpret("float32", T.shift_left(T.Cast("uint32", A_1[i]), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.Cast("uint32", B_1[i]), T.uint32(16)))
        # 移除了T.reinterpret("bfloat16", ...)，直接写入uint16
        D_1[i] = T.Cast("uint16", T.shift_right(T.reinterpret("uint32", T.exp(C[i])) + (T.bitwise_and(T.shift_right(T.reinterpret("uint32", T.exp(C[i])), T.uint32(16)), T.uint32(1)) + T.uint32(32767)), T.uint32(16)))
```

# LowerDeviceKernelLaunch

## 作用

LowerDeviceKernelLaunch Pass 用于将设备kernel启动操作lowering为具体的runtime API调用。该Pass处理跨设备或跨target的函数调用，将其转换为适合的调用约定。

该Pass执行以下操作：
1. **识别跨设备调用**：检测从host函数（如LLVM target）调用device kernel（如CUDA target）的情况
2. **收集kernel启动参数**：从kernel函数体中提取线程配置信息（如`threadIdx.x`、`blockIdx.x`等）和动态共享内存大小
3. **重写调用点**：将直接函数调用转换为`T.call_packed()`调用，并传递额外的启动参数
4. **更新kernel属性**：为kernel函数添加必要的属性，包括`calling_conv`、`tir.kernel_launch_params`、`tir.is_global_func`等

该Pass是通用的，支持各种异构计算场景（CPU调用GPU、不同codegen间调用等）。对于相同target内的调用，不做处理，由后端codegen处理为内部子程序调用。

## 效果

Pass将跨设备kernel调用转换为runtime packed function调用：

1. **调用点转换**：
   - 从`mod.kernel(A.data)`转换为`T.call_packed("kernel", A.data, 16)`
   - 添加kernel启动参数（如线程数16）到调用参数列表

2. **Kernel函数属性更新**：
   - 添加`"calling_conv": 2`（表示`kDeviceKernelLaunch`调用约定）
   - 添加`"tir.kernel_launch_params": ["threadIdx.x"]`（列出启动参数）
   - 添加`"tir.is_global_func": True`（标记为全局可见函数）
   - 保留或添加`"global_symbol"`（用于runtime查找kernel）

3. **参数传递**：
   - 原始函数参数：`A.data`
   - 额外启动参数：从kernel的`T.launch_thread("threadIdx.x", 16)`提取出的线程数`16`

4. **跨codegen调用处理**：
   - 同设备不同codegen（如LLVM调C）：转换为`T.call_extern()`
   - 同target内调用：保持不变，由codegen处理

从示例可以看出，Pass自动识别了kernel中的`T.launch_thread("threadIdx.x", 16)`，提取线程数16作为启动参数，并更新了调用点和kernel属性。

## 调用该Pass前IR

```python
@T.prim_func
def kernel(A: T.handle("float32", "global")):
    T.func_attr({"target": T.target({"arch": "sm_50", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
    A_1 = T.decl_buffer((16,), data=A)
    i = T.launch_thread("threadIdx.x", 16)
    A_1[i] = T.float32(0.0)

@T.prim_func
def main(A: T.Buffer((16,), "float32")):
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    # 直接调用kernel函数（跨设备：CPU -> GPU）
    Module.kernel(A.data)
```

## 调用该Pass后IR

```python
@T.prim_func
def kernel(A: T.handle("float32", "global")):
    # 添加了kernel启动所需的属性
    T.func_attr({"calling_conv": 2, "target": T.target({"arch": "sm_50", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["threadIdx.x"]})
    A_1 = T.decl_buffer((16,), data=A)
    i = T.launch_thread("threadIdx.x", 16)
    A_1[i] = T.float32(0.0)

@T.prim_func
def main(A: T.Buffer((16,), "float32")):
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    # 转换为call_packed，添加线程数参数16
    T.call_packed("kernel", A.data, 16)
```

# LowerTVMBuiltin

## 作用

LowerTVMBuiltin Pass 用于将TVM的内置函数（builtin functions）lowering为更底层的runtime API调用。该Pass处理多种TVM内置操作，将高层抽象转换为具体的运行时实现。

该Pass执行以下操作：
1. **设备内存分配**：将`T.allocate()`转换为`TVMBackendAllocWorkspace()`/`TVMBackendFreeWorkspace()`调用
   - 对于CPU小内存分配（< 1MB）保持原样，由codegen处理栈分配
   - 对于设备内存（GPU等）或大内存分配，转换为运行时API调用
2. **错误处理**：插入`T.tvm_throw_last_error()`调用检查内存分配和释放错误
3. **Packed函数调用**：将`T.call_packed()`转换为`tvm_call_packed_lowered()`，设置参数栈
4. **数组构造**：处理`tvm_stack_make_array()`和`tvm_stack_make_shape()`，分配栈空间并设置DLTensor结构
5. **设备特定操作**：处理DMA操作等设备特定builtin

该Pass是通用的，支持所有目标平台，但根据设备类型（CPU vs GPU）会有不同的lowering策略。

## 效果

Pass将设备内存分配转换为runtime workspace API调用：

1. **内存分配转换**：
   - Before: `ptr = T.allocate([16], "float32")` - 高层内存分配
   - After: `ptr = T.TVMBackendAllocWorkspace(2, 0, T.uint64(64), 2, 32)` - runtime API调用
   - 参数：设备类型(2=CUDA), 设备ID(0), 字节数(64), dtype_code(2=float), dtype_bits(32)

2. **错误检查插入**：
   - 分配后检查：`if T.isnullptr(ptr): T.tvm_throw_last_error()`
   - 释放后检查：`if T.TVMBackendFreeWorkspace(...) != 0: T.tvm_throw_last_error()`

3. **存储对齐**：
   - 添加`T.attr(ptr, "storage_alignment", 64)`属性，确保内存对齐

4. **内存释放**：
   - 在作用域结束时自动插入`TVMBackendFreeWorkspace()`调用

5. **设备类型判断**：
   - CPU小分配（< kMaxStackAlloca）：保持`T.allocate()`，由LLVM codegen处理栈分配
   - GPU或大内存分配：转换为workspace API

从示例可以看出，Pass将CUDA设备（device_type=2）上的16个float32分配（64字节）转换为TVMBackendAllocWorkspace调用，并添加了空指针检查和释放逻辑。

## 调用该Pass前IR

```python
@T.prim_func
def main():
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    T.attr("dummy", "device_type", 2)  # CUDA设备
    T.attr("dummy", "device_id", 0)
    # 高层内存分配
    buf = T.decl_buffer((16,))
    buf[0] = T.float32(0.0)
```

## 调用该Pass后IR

```python
@T.prim_func
def main():
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    # 转换为runtime workspace分配API
    buf: T.handle("float32", "global") = T.TVMBackendAllocWorkspace(2, 0, T.uint64(64), 2, 32)
    # 设置内存对齐
    T.attr(buf, "storage_alignment", 64)
    # 检查分配是否成功
    if T.isnullptr(buf):
        T.tvm_throw_last_error()
    buf_1 = T.decl_buffer((16,), data=buf)
    buf_1[0] = T.float32(0.0)
    # 释放内存并检查错误
    if T.TVMBackendFreeWorkspace(2, 0, buf) != 0:
        T.tvm_throw_last_error()
```

# LowerCustomDatatypes

## 作用

LowerCustomDatatypes Pass 用于将自定义数据类型的操作lowering为标准数据类型的操作。TVM允许用户注册自定义数据类型（如特殊的定点格式、posit数等），该Pass负责将使用这些自定义类型的操作转换为使用标准整数/浮点类型的等价操作。

**注意：该Pass没有专门的测试文件，以下分析基于源代码实现。**

该Pass的工作机制：
1. **识别自定义数据类型**：检查IR中所有表达式和语句，识别使用了已注册自定义数据类型的节点
2. **查找lowering函数**：对于每个使用自定义类型的操作，从全局注册表中查找对应的lowering函数
3. **调用lowering函数**：使用用户注册的lowering函数将自定义类型操作转换为标准类型操作
4. **变量和Buffer重映射**：更新变量和buffer的类型从自定义类型到标准类型（通常是相同位宽的uint类型）

支持的操作类型：
- **类型转换（Cast）**：自定义类型与其他类型之间的转换
- **算术运算**：Add, Sub, Mul, Div, Mod
- **比较运算**：EQ, NE, LT, LE, GT, GE
- **Min/Max运算**
- **内存分配（Allocate）**：自定义类型buffer分配，转换为uint类型
- **Buffer访问（BufferLoad/Store）**：重映射buffer类型
- **Intrinsic调用**：如sqrt、sigmoid等
- **立即数（FloatImm）**：自定义类型常量

该Pass是通用的，支持所有目标平台，但用户需要为每个target单独注册lowering函数。

## 效果

Pass通过查找用户注册的lowering函数，将自定义数据类型操作转换为标准类型操作：

1. **内存分配转换**（源码89-106行）：
   - Before: `ptr = T.allocate([100], "custom_float8")`
   - After: `ptr = T.allocate([100], "uint8")` - 转换为相同位宽的uint类型
   - 变量重映射：更新buffer_var的类型注解

2. **Buffer重映射**（源码138-164行）：
   - Before: `buf = T.decl_buffer((100,), "custom_float8")`
   - After: `buf = T.decl_buffer((100,), "uint8")`
   - 保持buffer形状和其他属性不变

3. **算术运算lowering**（源码210-227行，通过宏定义）：
   - Before: `result = a + b` (custom_float8类型)
   - After: `result = custom_float8_add_lowered(a, b)` - 调用用户注册的函数
   - 用户需要通过`datatype.register_op()`注册如`tvm.datatype.lower.llvm.Add.custom_float8`

4. **类型转换lowering**（源码47-61行）：
   - Before: `val = T.cast("custom_float8", x)`
   - After: 调用注册的cast lowering函数，如`tvm.datatype.lower.llvm.Cast.custom_float8.float32`

5. **Intrinsic调用lowering**（源码192-207行）：
   - Before: `result = T.sqrt(x)` (x为custom_float8)
   - After: 调用注册的intrinsic lowering，如`tvm.datatype.lower.llvm.Call.intrin.sqrt.custom_float8`

从源代码可以看出，Pass遍历所有可能包含自定义类型的节点，对每个节点检查其dtype的type_code是否已注册为自定义类型，如果是则查找并调用对应的lowering函数。

## 调用该Pass前IR

N/A

## 调用该Pass后IR

N/A

# LowerIntrin

## 作用

LowerIntrin Pass 用于将高层intrinsic操作lowering为目标平台可以理解的底层表示。该Pass处理多种高层操作，将它们转换为更接近硬件的指令或操作序列。

该Pass执行以下操作：
1. **FloorDiv/FloorMod优化**：将floor除法和取模转换为更高效的实现
   - 对于2的幂次方除数，使用位移和位掩码操作
   - 对于正除数，转换为truncdiv/truncmod
   - 对于一般情况，使用分析器优化选择最佳lowering策略
2. **FMA（Fused Multiply-Add）识别**：识别`a * b + c`模式并转换为FMA指令（如果目标支持）
3. **Target-specific intrinsic lowering**：根据注册的lowering规则转换平台特定的intrinsic
4. **Broadcast/Cast优化**：重排broadcast和cast顺序以生成更高效的向量指令

该Pass是通用的，支持所有目标平台，但会根据target属性（如LLVM、CUDA、WebGPU等）和架构（如AArch64）应用不同的lowering规则。

## 效果

Pass将高层数学操作转换为高效的底层实现：

1. **FloorDiv优化**（源码98-192行）：
   - **2的幂次方优化**：
     - Before: `B[i] = A[i] // 8` (floordiv)
     - After: `B[i] = T.shift_right(A[i], 3)` (右移3位，因为8 = 2^3)
   - **正除数优化**：当分析器能证明除数为正时，转换为truncdiv
   - **符号未知优化**：使用条件判断或位运算实现正确的floor语义

2. **FloorMod优化**（源码194-269行）：
   - **2的幂次方优化**：
     - Before: `B[i] = A[i] % 8` (floormod)
     - After: `B[i] = T.bitwise_and(A[i], 7)` (与掩码0b111按位与)
   - **正除数优化**：转换为truncmod
   - **一般情况**：使用条件判断实现正确的floor语义

3. **FMA识别**（源码354-368行）：
   - Before: `result = a * b + c` (展开的乘加)
   - After: `result = T.call_intrin("float32", "llvm.fma", a, b, c)` (FMA intrinsic)
   - 条件：目标支持FMA且为浮点类型

4. **Target-specific lowering**（源码67-86行）：
   - 查找并应用注册的lowering函数
   - 查找顺序：`<target>.FLowerIntrinsic` → `<target>.FLegalize` → `default.FLowerIntrinsic` → `default.FLegalize`
   - 对于AArch64，额外查找`llvm.aarch64.FLowerIntrinsic`

5. **Broadcast/Cast优化**（源码320-352行）：
   - Before: `broadcast(cast(x))` (先cast再broadcast)
   - After: `cast(broadcast(x))` (先broadcast再cast，对于widening cast)
   - 目的：生成更高效的SIMD指令（如ARM的vmla vs vmlal）

从示例可以看出，Pass将除以8的floordiv转换为右移3位，将模8的floormod转换为与7按位与，这些都是高效的位运算。

## 调用该Pass前IR

```python
@T.prim_func
def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    for i in range(16):
        # 使用高层的floordiv和floormod操作
        B[i] = A[i] // 8 + A[i] % 8
```

## 调用该Pass后IR

```python
@T.prim_func
def main(A: T.Buffer((16,), "int32"), B: T.Buffer((16,), "int32")):
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    for i in range(16):
        # 转换为高效的位运算
        # floordiv(A[i], 8) → shift_right(A[i], 3)
        # floormod(A[i], 8) → bitwise_and(A[i], 7)
        B[i] = T.shift_right(A[i], 3) + T.bitwise_and(A[i], 7)
```

# LowerDeviceStorageAccessInfo

## 作用

LowerDeviceStorageAccessInfo Pass 用于lowering特殊设备存储作用域的访问。该Pass处理具有特定存储作用域（storage scope）的buffer分配，根据内存信息（MemoryInfo）决定如何访问这些特殊的存储区域。

该Pass执行以下操作：
1. **识别特殊存储作用域**：检查Allocate节点的storage scope标签（tag）
2. **查找内存信息**：为每个特殊作用域获取对应的MemoryInfo（通过注册的函数）
3. **转换分配方式**：
   - **CPU可访问的存储**（有head_address）：将Allocate转换为LetStmt，使用head_address初始化指针
   - **CPU不可访问的存储**（无head_address）：移除Allocate节点（CPU侧不需要访问）
4. **处理tvm_access_ptr**：转换buffer访问指针为适合目标存储的格式

该Pass主要用于特殊硬件架构（如Hexagon的VTCM、GPU的纹理内存等），是平台特定的存储管理机制。对于标准存储（如全局内存、共享内存），该Pass不做处理。

## 效果

Pass根据存储作用域的内存信息转换buffer分配：

1. **CPU可访问存储的转换**（源码48-56行）：
   - Before: `ptr = T.allocate([16], "float32", scope="global.test_with_head_address")`
   - After: `ptr = T.call_extern("handle", "dummy_head_address")` - 使用LetStmt替换Allocate
   - 说明：存储区域有固定的访问地址（head_address），通过该地址访问

2. **DeclBuffer更新**（源码67-73行）：
   - Before: `buf = T.decl_buffer((16,), scope="global.test_with_head_address")`
   - After: `buf = T.decl_buffer((16,), data=buf, scope="global.test_with_head_address")`
   - 说明：更新DeclBuffer的data字段为head_address，避免悬空引用

3. **CPU不可访问存储的移除**（源码48-56行）：
   - Before: `ptr = T.allocate([16], "float32", scope="global.test_without_head_address")`
   - After: (移除Allocate，只保留body)
   - 说明：CPU无法直接访问该存储（如GPU纹理内存），CPU侧引用应已被lowered

4. **tvm_access_ptr转换**（源码82-114行）：
   - 对于有MemoryInfo的buffer，调用`MakeTaggedAccessPtr()`
   - 根据`info->unit_bits`和数据类型计算正确的offset
   - 对于非handle类型的指针，转换为索引形式

从示例可以看出，Pass将带有特殊storage scope的Allocate转换为调用注册的head_address函数（如`dummy_head_address`），这使得可以访问预分配的硬件特定内存区域。

## 调用该Pass前IR

```python
@T.prim_func
def main():
    # 使用特殊存储作用域的buffer（CPU可访问）
    buf = T.decl_buffer((16,), scope="global.test_with_head_address")
    T.evaluate(buf.data)
```

## 调用该Pass后IR

```python
@T.prim_func
def main():
    # Allocate被替换为调用head_address函数
    buf: T.handle("float32", "global.test_with_head_address") = T.call_extern("handle", "dummy_head_address")
    # DeclBuffer更新data字段
    buf_1 = T.decl_buffer((16,), data=buf, scope="global.test_with_head_address")
    T.evaluate(buf)
```

# CombineContextCall

## 作用

CombineContextCall Pass 用于优化设备上下文的重复调用。当多次调用`tvm_thread_context(device_context(...))`且参数相同时，该Pass会将这些调用合并，将相同的context调用提取到变量中缓存，避免重复的runtime调用开销。

**适用范围：通用Pass，适用于所有使用设备上下文的异构计算场景。**

## 效果

Pass识别对`tvm_thread_context`的重复调用，将相同参数的`device_context(...)`调用提升到函数开头，并用缓存变量替代后续的重复调用。这减少了runtime函数调用次数，提升执行效率。

## 调用该Pass前IR

```python
@T.prim_func
def main(dev_type: T.int32, n: T.int32):
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    A = T.allocate([n], "float32", "global")
    for i in range(n):
        # 多次调用device_context，参数为(dev_type, 0)
        T.call_extern("int32", "fadd", T.tvm_thread_context(T.call_extern("handle", "device_context", dev_type, 0)), A)
        for j in range(10):
            # 调用device_context，参数为(dev_type, 1)
            T.call_extern("int32", "fadd", T.tvm_thread_context(T.call_extern("handle", "device_context", dev_type, 1)), A)
            # 再次调用device_context，参数为(dev_type, 0) - 重复！
            T.call_extern("int32", "fadd", T.tvm_thread_context(T.call_extern("handle", "device_context", dev_type, 0)), A)
```

## 调用该Pass后IR

```python
@T.prim_func
def main(dev_type: T.int32, n: T.int32):
    T.func_attr({"target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-conda-linux-gnu", "tag": ""})})
    # 相同的device_context调用被提取到变量中缓存
    ctx_cache_: T.handle = T.call_extern("handle", "device_context", dev_type, 0)
    ctx_cache__1: T.handle = T.call_extern("handle", "device_context", dev_type, 1)
    A = T.allocate([n], "float32", "global")
    for i in range(n):
        # 直接使用缓存的context变量
        T.call_extern("int32", "fadd", ctx_cache_, A)
        for j in range(10):
            T.call_extern("int32", "fadd", ctx_cache__1, A)
            T.call_extern("int32", "fadd", ctx_cache_, A)
```

# LowerWarpMemory

## 作用

LowerWarpMemory Pass 用于将warp级别的内存（scope="warp"）lowering为local内存加上warp shuffle指令。在GPU中没有实际的"warp memory"，该Pass将warp内存抽象转换为线程本地内存，并通过warp shuffle指令在warp内的线程间传递数据。

**适用范围：GPU特定Pass，需要target具有warp支持（如CUDA、ROCm），典型warp size为32。**

**注意：该Pass没有专门的测试文件，以下分析基于源代码实现（src/tir/transforms/lower_warp_memory.cc）。**

## 效果

Pass将warp作用域的内存分配和访问转换为：
1. **内存分配**：将`warp_mem[n * width * m]`转换为`local_mem[n * m]`（size缩小为1/width）
2. **存储操作**：将`warp_mem[m * warp_index + ...]`转换为`local_mem[m * y + x]`（去除warp_index维度）
3. **加载操作**：将`warp_mem[m * z + ...]`转换为`warp_shuffle(local_mem[m * y + x], z)`

关键转换规则（来自源码注释）：

**转换前**：
```
alloc warp warp_mem[n * width * m]
store warp_mem[m * warp_index + (width * m) * y + x]
load warp_mem[m * z + (width * m) * y + x]
```
其中 x ∈ [0, m), y ∈ [0, n), width = threadIdx.x的范围（≤ warp_size）

**转换后**：
```
alloc local local_mem[n * m]
store local_mem[m * y + x]
warp_shuffle(load local_mem[m * y + x], z)
```

## 调用该Pass前IR

N/A

## 调用该Pass后IR

N/A
