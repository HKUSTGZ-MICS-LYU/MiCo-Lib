# BNCFU 预打包 W8A8 / W4A8：Q2 与 Q1.58 两种硬件

## 统一接口与模式选择

[MiCo_Q8_MatMul](bitnet_cfu_q8_matmul.c) 直接计算 Python 预打包权重：

```c
void MiCo_Q8_MatMul(int32_t *O, const Tensor2D_Q8 *x, const Tensor2D_Q8 *w);
```

| 构建配置 | Python 参数 | 权重格式 | 最低硬件需求 | W8/W4 平面数 |
| --- | --- | --- | --- | --- |
| `BNCFU_WITH_Q2=1` | `--qtype 2b` | `bncfu-radix4-v1` | 真 INT2，含 -2 | 4 / 2 |
| `BNCFU_WITH_Q2=0`（默认） | `--qtype 1.5b` | `bncfu-ternary3-v1` | Q1.58，只有 -1/0/+1 即可 | 6 / 3 |

Q1.58 在硬件 CLI 中写作 `1.5b`。模式在编译期选择，**两种产物不可互换**；
切换 `BNCFU_WITH_Q2` 后须用对应 Python 参数重新生成权重，不会自动识别数据。
Python 为保留旧产物兼容性默认 `--qtype 2b`，因此建议始终显式指定模式。

- `x->shape=[M,K]`，`x->data` 为普通连续 signed INT8 activation。
- `w->shape=[N,K]` 保留逻辑维度；**`w->data` 是对应模式、bits=8、匹配 VLEN 的
  Python 产物**，按 `B=VLEN/8` 字节对齐，具备 metadata 指定的完整可读长度。
- `O` 为连续 INT32 `[M,N]`，精确计算原始权重的 `X * W^T`。
- Tensor 定义、签名、scale/wq 字段不变，但这些字段不能识别 layout、实际字节
  容量或 VLEN。加载侧必须保证匹配；不能向此覆盖入口传入 raw INT8 权重。
- 其它 target 和 `MiCo_Q8_MatMul_Ref` 仍消费原始 INT8 权重，比较时使用独立 raw W。
- 不做 runtime 权重重编码、权重复制、堆分配、隐式指针缓存或 raw scalar fallback。
  Library 没有 C 打包函数；同一只读产物可重复使用，权重改变时在外部重新打包。
- 不在这里应用 scale、bias、requantization。非对齐 activation/尾块仍使用栈 scratch。

`TARGET=vexii_soc OPT=bncfu` 自动覆盖 Q8；`OPT="bncfu unroll"` 中 BNCFU Q8
优先，unroll Q4 保留。`USE_ALT_LAYOUT` 对非空 K 报错。空 M/N 为 no-op，K=0
写零，两种模式均不访问 CFU。Q2T/Q8 量化硬件不是依赖，没有修改硬件源码。

## Q2：radix-4（保留原格式）

```text
u = w + 2^(bits-1)
dp = ((u >> (2*p)) & 3) - 2
W8 = 42 + d0 + 4*d1 + 16*d2 + 64*d3
W4 =  2 + d0 + 4*d1
```

硬件码 `00=0,01=+1,10=-2,11=-1`。此路径需要已有
`--bitnet-cfu-with-q2`；CONFIG 无法启用未生成的 -2 硬件。

格式 `bncfu-radix4-v1`，`P=bits/2`、`T=4/P`：

```text
[N][ceil(K/(B*T)) groups][T tiles][P planes][B/4 bytes]
size = N * ceil(K/(B*T)) * B
```

每字节四个 lane，低 lane 在低两位，低位平面在前。padding 编码原始零，不是零 digit。
VLEN256 的 W8 一个 32B 向量是 `[P0 8B][P1 8B][P2 8B][P3 8B]`；W4 是
`[tile0.P0 8B][tile0.P1 8B][tile1.P0 8B][tile1.P1 8B]`。

[Q2 核心/API](bitnet_cfu_radix4.h) 保留大小查询及预打包 GEMM/GEMV。
每行 activation correction 只计算一次，在全部输出通道间共享。

## 仅 Q1.58：平衡三进制（新增）

精确表示而非低精度近似量化：

```text
w = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4 + 243*d5   # W8
w = d0 + 3*d1 + 9*d2                            # W4
dp in {-1,0,+1}
```

六位覆盖 [-364,364]，足以表示全部 W8；三位覆盖 [-13,13]，足以表示全部 W4。
Python 从低位提取 `d=(w+1)%3-1; w=(w-d)//3`，正确处理负数。码为
`-1=>11,0=>00,+1=>01`，不使用 `10`，因此没有 -2 路径也能精确计算。

计算为 `Y=sum(3^p * BDOT(A,d_p))`，**不需要 42/2 correction 或 activation sum 扫描**。
[三进制内核](bitnet_cfu_ternary3.c) 保留 INT64 分平面累加和延迟重构；
[API](bitnet_cfu_ternary3.h) 同样只含大小查询与 GEMM/GEMV，无 C 打包器。

### 连续 plane-slot 格式

`bncfu-ternary3-v1`，W8 的 `P=6`，W4 的 `P=3`：

```text
tiles = ceil(K/B)
slots = tiles * P
row_vectors = ceil(slots/4)
size = N * row_vectors * B
layout: [N][row_vectors][4 plane slots][B/4 bytes]
slot s = tile*P + plane
vector = s/4, slot_in_vector = s%4
```

相邻 tile 共用尚未用完的向量槽位，避免每个 tile 单独补满 4/8 个平面：

```text
W8:
vector0: [T0.P0][T0.P1][T0.P2][T0.P3]
vector1: [T0.P4][T0.P5][T1.P0][T1.P1]
vector2: [T1.P2][T1.P3][T1.P4][T1.P5]

W4:
vector0: [T0.P0][T0.P1][T0.P2][T1.P0]
vector1: [T1.P1][T1.P2][T2.P0][T2.P1]
vector2: [T2.P2][T3.P0][T3.P1][T3.P2]
```

每槽 B/4 bytes，字节内 lane 顺序与 Q2 相同。最后 K 尾 lane 与最终未用槽均为零。
W8 一个 tile 用两个向量，两个 tile 共三个；W4 一个 tile 用一个，四个共三个。

调度中 A 始终在 bank0、W 在 bank1..depth-1：每 tile 只 LOAD A 一次，仅在
`slot%4==0` 时 LOAD 新 W 向量；跨 tile 的半个向量保留 W 游标，不能重置。
这依赖现有 LOAD 只重置目标 bank 游标的语义，无需硬件改动。

主体权重空间为 W8 12 bits/weight、W4 6 bits/weight，约为 Q2 格式的 1.5 倍；
**尾部对齐按上式计费，短行可能不是 1.5 倍**。每 tile BDOT 为 W8 六条、W4 三条，
分别比 Q2 四/两条多 50%，但指令数不等于周期数，也不保证固定性能比例。

## Python 预处理

从 VexiiMico 根目录执行，先激活环境；[工具](weight_recode.py) 只使用标准库：

```bash
source /home/jzj/miniconda3/etc/profile.d/conda.sh
conda activate mico_env

# 仅 Q1.58 硬件：对应 BNCFU_WITH_Q2=0
python -B sw/MiCo-Lib/targets/vexii_soc/bncfu/weight_recode.py weights_i8.bin \
  --rows 16 --cols 512 --bits 8 --vlen 256 --qtype 1.5b \
  --output weights_ternary3.bin --metadata weights_ternary3.json

# 有真 Q2 硬件：对应 BNCFU_WITH_Q2=1，旧字节格式不变
python -B sw/MiCo-Lib/targets/vexii_soc/bncfu/weight_recode.py weights_i8.bin \
  --rows 16 --cols 512 --bits 8 --vlen 256 --qtype 2b \
  --output weights_radix4.bin --metadata weights_radix4.json
```

Python API：`pack_weights(values,bits=8,vlen=256,qtype="2b")`，
`unpack_weights(blob,rows,cols,bits,vlen,qtype="2b")`。
输入为 row-major signed unpacked bytes。INT4 使用 `--bits 4`，仍每个值一个 byte，
不是 nibble。输出无二进制头，JSON 包含
`format/qtype/bits/planes/rows/cols/vlen/row_bytes/layout`。
三进制 decoder 拒绝非法码 2、非零 padding 和解码超出逻辑范围的数据；
**C 计算路径假设产物合法，不逐字节扫描或自动检测格式**。
W4 显式 API 不改变现有 `MiCo_Q8x4_MatMul` 的 nibble 契约。

## 数值、内存与硬件约束

- VLEN 必须匹配；物理 bank 数至少为 `BNCFU_REG_DEPTH`（2..8，建议完全匹配）。
- 每条完整 BDOT 处理 `VLEN/8` 个 activation，MACLEN 只影响内部切片/周期。
- 推荐同步 RF，调用方独占 CFU；Q2 内核结束为 2b，三进制结束为 1.5b。
- 两种模式均 INT64 累加、延迟重构、检查 INT32 范围，非空计算 K<=INT32_MAX。
- packed W 按 B 字节对齐，完整可读且含 padding；输入输出不得重叠。
- activation 非对齐或尾块复制到对齐栈 scratch 补零后才进行整向量 LOAD。
- 初始 DMA fence 发布 X/W，scratch 改写后再次发布，直至 BDOT 完成才重用。
  依赖当前 Vexii 缓存维护语义，不宣称通用 `fence.i` 能清理任意平台数据缓存。
- 大小及维度溢出均检查；显式 API 返回共同 `OK/EINVAL/ENOSPC/ERANGE`。
  ERANGE 时之前的输出可能已写入，统一入口使用 `MiCo_assert` 报错。
- 支持 power-of-two VLEN32..512 的软件格式，但真实硬件仍需满足总线宽度约束。

## 验证命令

所有测试先激活 `mico_env`。主机模型不能替代真实 RTL/缓存/总线时序。

```bash
python -B -m unittest discover \
  -s sw/MiCo-Lib/targets/vexii_soc/bncfu -p test_recode.py -v
python -B sw/tests/test_bncfu_radix4_host.py
```

主机 6 组回归已通过，含双模式 14 个 VLEN/bank ASan/UBSan 配置、全部 W8/W4
与全部 INT8 activation 的单元素组合、随机 GEMM/GEMV、尾块/非对齐 X、准确的
BDOT/W LOAD/A LOAD 计数、统一入口只读复用与零堆分配、正负溢出、大 K、空维度、
ALT 拒绝、unroll 优先级和两种 CONFIG 来回切换。无 Q2 的模型禁止 CONFIG2，
确保该分支没有借用 -2；所有 fixture 都由 Python 生成，并绑定编译时模式与 VLEN。

### 仅 Q1.58 的 SoC 编译/仿真

```bash
python -B sw/tests/gen_bncfu_radix4_fixtures.py --vlen 256 --qtype 1.5b \
  --output sw/build_bncfu_q8_ternary3/bncfu_radix4_fixtures.h
make -C sw -f Makefile compile \
  MAIN=tests/bncfu_radix4_test BUILD=build_bncfu_q8_ternary3 \
  EXTRA_INCLUDES=build_bncfu_q8_ternary3 \
  TARGET=vexii_soc MARCH=rv32imafc_zifencei OPT="bncfu ref" SPRAM=1 \
  VLEN=256 BNCFU_REG_DEPTH=5 BNCFU_WITH_Q2=0 BNCFU_QUANT_WIDTH=128
BNCFU_WITH_Q2T=0 BNCFU_RF_SYNC=1 \
  bash sim_soc_bncfu.sh sw/tests/bncfu_radix4_test.elf 1.5b 256 128 5 0 1 128 0
```

该仿真不传 `--bitnet-cfu-with-q2`，不是在 Q2 硬件上仅修改默认 qtype。
Q2 版使用新的 BUILD 目录，fixture 参数改 `--qtype 2b`，编译设置
`BNCFU_WITH_Q2=1`，sim 参数改为 `2b 256 128 5 1 1 128 0`。
勿并发运行两个仿真（共享 simWorkspace），也不要用旧 BUILD 复用其它参数的 object。

## 验证结果与性能

三进制核心通过 VLEN32/64/128/256/512 × depth2/5/8 的 15 组严格 RV32 编译；
无 FPU 的 `rv32imac_zicsr_zifencei` 也通过。本工具链编译 CFU enable 需要显式
Zicsr。Q1.58 SoC 完整编译通过，链接确认只有 ternary3 计算内核，无 radix4 或 C 打包器。
Python 编码/CLI 回归 23 项通过（包含保留的 10 项 Q2 测试和三进制独立 golden、
穷举、游标/补零、随机 dot、metadata 及非法码/非零 padding 拒绝测试）。
Q2/unroll 的完整 RISC-V 回归链接也通过：统一入口唯一强符号，选择 radix4，
不链接 ternary3 内核；两种模式均确认没有 C 打包器。

### Q1.58-only RTL（2026-10-19）

VLEN256/MACLEN128/depth5、同步 RF、compute pipe、bus64、seed42；Q2/Q2T/Q8
均未生成。Python 产物驱动的 W4/W8、尾块、多输出通道、统一入口和重复只读调用
**全部 PASS**。`M=1,N=16,K=512` 的同次仿真实测：

| Q1.58 路径 | 测试标量循环 | 被测路径 cycles | 相对参考 |
| --- | ---: | ---: | ---: |
| W4A8 显式三进制内核 | 76,673 | 15,795 | 4.85x |
| W8A8 显式三进制内核 | 76,622 | 28,612 | 2.68x |
| `MiCo_Q8_MatMul` 三进制入口 | 76,622 | **28,783** | **2.66x** |

尾块/小矩阵仍可能慢于标量，例如 M3/N3/K257 统一入口 23,227 cycles，对照
21,327 cycles；M1/N1/K65 则为 2,817 对 717 cycles。不能泛化为全形状加速。

### Q2 已完成基线（历史测量）

已完成的 Q2 RTL 基线：VLEN256/MACLEN128/depth5、同步 RF、compute pipe、seed42，
W4/W8 与统一入口全部 PASS。`M=1,N=16,K=512` 的历史实测 cycles：

| Q2 路径 | 测试标量循环 | 被测路径 | 相对参考 |
| --- | ---: | ---: | ---: |
| W4A8 显式内核 | 76,678 | 13,975 | 5.49x |
| W8A8 显式内核 | 76,671 | 21,992 | 3.49x |
| `MiCo_Q8_MatMul` | 76,671 | 21,942 | 3.49x |

参考是含 INT64 和范围检查的测试循环，不是最佳 INT8 SIMD，也不是 REF 函数的计时。
计时排除离线 Python 预处理，不代表整模型端到端加速；小矩阵/尾块仍可能慢于标量。
包装入口与底层的缓存/调用上下文不同，微小计时差不表示包装层优于底层。
先前运行时打包的 303,271-cycle 路径已经删除。尚未验证大型模型加载、V2 RTL、
其它 RF/总线配置、面积或功耗。
