# 实验性代码重构 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在没有机器狗(Go2)与 Lekiwi 连接的空窗期，把仓库里"实验性/历史性"代码整理到可维护状态：修复阻断工作站测试的跨边界依赖，把废弃的硬件直连代码隔离为明确的 rollback 路径，参数化 PowerShell 启动脚本中的硬编码，并在不改变上层语义与安全边界的前提下保证所有软件测试通过。

**Scope 边界（本轮不做）:**
- 不新增/修改任何 Go2、Lekiwi、Piper 的运动能力或安全仲裁语义。
- 不改 `ubrobot_manipulation`（ROS ament 包）的对外行为；只允许调整其纯 Python `authority.py` 的导入来源（re-export），保持其 API 不变。
- 不动 `deploy/` 下容器/系统启动逻辑与 `docs/validation` 历史报告。
- 不提交任何 token、IP 序列号、机密。

**原则:**
- `robot_edge` / `ubrobot_contracts` 是纯 Python 工作站可测层，**不得 import 任何 ROS ament 包**（plan 已有不变量：workstation 测试不导入 `unitree_sdk2py`/`piper_sdk`/`rclpy`）。
- 废弃硬件直连代码保留为显式 rollback 路径，但必须 **lazy-import + 明确降级**，不能因 `import` 副作用在无硬件时崩溃。
- 所有改动以 `python -m unittest` 全量通过为验收门槛。

---

## 现状盘点（2026-08-08 已核实）

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 1 | `robot_edge/motion_arbitration.py:37` `from ubrobot_manipulation.authority import AuthorityTracker` | `src/robot_edge/motion_arbitration.py` | 工作站 `tests/robot_edge/test_motion_arbitration.py` import 失败（`ModuleNotFoundError: ubrobot_manipulation`），是 210 个 robot_edge 测试里唯一 1 个 error |
| 2 | 与模块 docstring 矛盾（"fully unit-testable on a workstation"） | 同上 | 依赖 `ros_depends_ws` 才会通过，CI/工作站路径不可信 |
| 3 | `arm_action.py` 630 行死代码：引用未定义的 `tf_trans`、`tf2_ros`、`do_transform_pose`、`Float64`、`self.piper_mp`、`self.target_pose_pub` | `src/ubrobot/robots/arm_action.py` | 无法 import，纯实验残留，无任何调用方 |
| 4 | `ubrobot.py`(Go2Manager)：`__init__` 无条件 `self.lekiwi_base.connect()`，硬编码相机序列号 `348522070565`，`from thread_utils import` 依赖脚本目录而非包 | `src/ubrobot/robots/ubrobot.py` | legacy 路径无硬件即崩溃；`pipeline.py` 的 `_LegacyBackend` 仍会构造它 |
| 5 | `unitree_go2_robot.py` 直接 `SportClient` 运动，plan 已声明废弃（Go2 运动一律 `/cmd_vel`） | `src/ubrobot/robots/unitree_go2_robot.py` | 误导性实验代码 |
| 6 | `lekiwi_base.py` 大段注释掉的 arm 代码 | `src/ubrobot/robots/lekiwi/lekiwi_base.py` | 可读性差 |
| 7 | `start_console_hardware.ps1` 硬编码 `EdgeHost`/`PiTokenPath`/token 缓存无新鲜度校验 | `scripts/start_console_hardware.ps1` | 机器 IP/路径变更即坏 |

---

## Task 1: 修复 `AuthorityTracker` 跨边界依赖（先写失败测试）

> **STATUS: DONE 2026-08-08.** robot_edge 241 测试全绿（此前 test_motion_arbitration 为唯一 error）；ROS 侧 authority.py 改为 re-export；CI 的 continue-on-error 已移除。

**Files:**
- Create: `src/ubrobot_contracts/motion_authority.py`（从 `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/authority.py` 平移纯 Python 逻辑）
- Modify: `src/robot_edge/motion_arbitration.py`（改 import 到 `ubrobot_contracts.motion_authority`）
- Modify: `ros_depends_ws/src/ubrobot_manipulation/ubrobot_manipulation/authority.py`（改为 re-export `ubrobot_contracts.motion_authority.AuthorityTracker`，保持 `from ubrobot_manipulation.authority import AuthorityTracker` 兼容）
- Create: `tests/robot_edge/test_motion_authority_contract.py`

**Step 1: 写失败测试**
- 断言 `robot_edge.motion_arbitration` 在仅 `PYTHONPATH=src` 时可 import（不再需要 `ros_depends_ws/src`）；
- 断言 `ubrobot_contracts.motion_authority.AuthorityTracker` 与 `ubrobot_manipulation.authority.AuthorityTracker` 行为一致（lease/stationary 判定）；
- 断言 `motion_arbitration.py` 源码不包含 `ubrobot_manipulation` import（fixture 保护，参考既有 `test_dependency_contract.py` 范式）。

Run: `python -m unittest tests.robot_edge.test_motion_arbitration tests.robot_edge.test_motion_authority_contract -v`
Expected: FAIL（`ubrobot_contracts.motion_authority` 尚不存在）。

**Step 2: 平移 + re-export**
- `motion_authority.py` 保持 `AuthorityTracker` 原逻辑（`LEASE_MAX_AGE_SEC=0.5`、`CMD_VEL_WINDOW_SEC=0.5`、`CMD_VEL_EPSILON=1e-4`）与 docstring，不引入 ROS。
- `authority.py`（ROS 侧）改为 `from ubrobot_contracts.motion_authority import AuthorityTracker` 并保留 `__all__`/常量 re-export，避免 ROS 包内 `grasp_object_server.py` 改动。

**Step 3: 验证并提交**
Run:
```powershell
$env:PYTHONPATH="src"; python -m unittest discover -s tests/robot_edge -p "test_*.py" -q
$env:PYTHONPATH="src;ros_depends_ws/src"; python -m unittest ros_depends_ws.src.ubrobot_manipulation.test.test_authority -v
git diff --check
git add src/ubrobot_contracts src/robot_edge ros_depends_ws/src/ubrobot_manipulation tests/robot_edge
git commit -m "refactor: move pure motion authority into ubrobot_contracts"
```

**Acceptance:** `tests/robot_edge` 210 个测试全绿；`ubrobot_manipulation.test_authority` 仍绿；motion_arbitration 不再依赖 ROS 包。

---

## Task 2: 隔离 `Go2Manager` / `UnitreeGo2Robot` legacy 硬件直连代码

> **STATUS: DONE 2026-08-08.** `Go2Manager.__init__` 不再连接硬件（显式 `connect_base()`）；`unitree_go2_robot` lazy import + DeprecationWarning；`_LegacyBackend` 明确报错；`lekiwi_base` 清理注释掉的 arm 代码并修复 calibrate/setup_motors 对未定义 `arm_motors` 的引用；修复 `ubrobot.py` 的 `thread_utils` 裸导入。

**Files:**
- Modify: `src/ubrobot/robots/ubrobot.py`（lazy 构造 + 显式 `connect_base()` 替代 `__init__` 副作用）
- Modify: `src/chat_ui/pipeline.py`（`_LegacyBackend` 改为捕获初始化失败并给出明确报错，而不是裸崩溃）
- Modify: `src/ubrobot/robots/unitree_go2_robot.py`（顶部加明确"已废弃，Go2 运动请走 /cmd_vel"警告 + lazy import）
- Modify: `src/ubrobot/robots/lekiwi/lekiwi_base.py`（删除被注释的 arm 代码块，仅保留 base 相关逻辑）
- Create: `tests/legacy/test_legacy_rollback_imports.py`

**Step 1: 写失败测试**
- `test_legacy_rollback_imports.py` 断言：`import ubrobot.robots.ubrobot` 不会触发硬件连接（`LeKiwi`/`CameraOdom` 构造只在显式调用时发生）；`unitree_go2_robot` 模块 import 时若 `unitree_sdk2py` 缺失应产生 DeprecationWarning 而非 ImportError；`pipeline.py` 在 `UBROBOT_CHAT_BACKEND=legacy` 且无硬件时给出可读错误。

Run: `python -m unittest tests.legacy.test_legacy_rollback_imports -v`
Expected: FAIL（当前 `ubrobot.py` import 即连硬件/可能崩溃）。

**Step 2: 实现隔离**
- `Go2Manager.__init__` 移除 `self.lekiwi_base.connect()` 副作用，改为 `connect_base()` 方法，并在 `start_threads()` 前显式调用；`agent_response` 保持原语义。
- `_LegacyBackend.__init__` 用 `try/except` 捕获并 re-raise 带诊断的 `RuntimeError`。
- `unitree_go2_robot.py` 顶部加 `warnings.warn(DeprecationWarning)`，`import unitree_sdk2py` 移到 `__init__` 内 try。

**Step 3: 验证并提交**
Run:
```powershell
$env:PYTHONPATH="src"; python -m unittest tests.legacy.test_legacy_rollback_imports -v
$env:PYTHONPATH="src"; python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -q
git diff --check
git add src/ubrobot src/chat_ui tests/legacy
git commit -m "refactor: isolate legacy hardware-direct rollback code"
```

**Acceptance:** `import ubrobot.robots.ubrobot` 在无硬件/无 SDK 环境不崩溃；legacy 路径错误信息可读；cortex_navigation 全绿。

---

## Task 3: 参数化与健壮化 `start_console_hardware.ps1`

**Files:**
- Modify: `scripts/start_console_hardware.ps1`
- Modify: `scripts/operator_console.ps1`（仅在必要处，抽取公共函数，避免重复）

**Step 1: 写验收脚本（不运行硬件）**
- 在 `scripts/hardware/` 新增 `test_start_console_hardware_params.ps1`（Pester/纯 PowerShell 均可）：断言默认参数可被 `-EdgeHost`/`-PiTokenPath` 覆盖、`status/logs/stop` 分支不触碰 token、`UBROBOT_EDGE_HARDWARE_AUTHORITY=true` 时启动会因 E-stop 未绑 fail-closed（模拟环境变量）。

Run: `powershell -File scripts/hardware/test_start_console_hardware_params.ps1`
Expected: FAIL（当前脚本无参数校验/无 fail-closed 单元可测性）。

**Step 2: 实现**
- token 缓存存在时校验 `LastWriteTime`（> 24h 则提示重新 scp 而非静默复用）；
- `EdgeHost`/`EdgePort`/`PiTokenPath` 已可参数化（已有），补充 `-ForceRefresh` 开关强制重新拉取；
- 校验 `scp`/`ssh` 存在；`UBROBOT_EDGE_HARDWARE_AUTHORITY` 与 `UBROBOT_EDGE_ESTOP_EXEMPTED` 组合时输出安全提示。

**Step 3: 验证并提交**
Run:
```powershell
powershell -File scripts/hardware/test_start_console_hardware_params.ps1
git diff --check
git add scripts
git commit -m "refactor: harden start_console_hardware script parameters"
```

**Acceptance:** 无硬件环境下脚本参数/分支可被测试覆盖；硬编码仍保留默认值但可覆盖。

---

## Task 4: 清理死代码与残留实验文件

> **STATUS: DONE 2026-08-08.** `arm_action.py` 移入 `archive/`（不再可 import）；`src/ubrobot/robots/README.md` 重写为状态表；新增 `docs/hardware/legacy-rollback.md`。

**Files:**
- Modify: `src/ubrobot/robots/arm_action.py`（整体判断：无调用方 → 移入 `examples/experimental_archive/` 或标记 `DEPRECATED - DO NOT USE` 头注释 + 顶层 `raise ImportError("archived")` ？—— 需在实现时与 owner 确认是否物理删除）
- Modify: `src/ubrobot/robots/README.md`（清理杂散链接，写清楚每个子目录的状态：可用/已废弃/实验）
- Create: `docs/hardware/legacy-rollback.md`（记录 Go2Manager/unitree_go2_robot/arm_action 的废弃状态与替代路径）

**Step 1: 盘点**
- 用 `rg -l "arm_action|PoseTransformer|RobotState" src tests` 确认 `arm_action.py` 无外部调用方（已核实：仅自引用）。
- 确认 `unitree_go2_robot.py` 被 `ubrobot.py` 引用但 import 在注释中（`#self.go2client = UnitreeGo2Robot()`），实际未构造。

**Step 2: 处理**
- 按 owner 决定：物理删除 vs 归档。默认建议：`arm_action.py` 移入 `docs/` 外的 `archive/`（不进 `src/`），避免 setuptools 打包坏模块。
- `README.md` 精简为状态表。

**Step 3: 验证并提交**
Run:
```powershell
$env:PYTHONPATH="src"; python -m unittest discover -s tests -q
git diff --check
git add -A
git commit -m "chore: archive dead experimental code and document rollback"
```

**Acceptance:** `src/ubrobot/robots` 不再包含无法 import 的模块；README 清晰标注状态。

---

## Task 5: 全量回归与文档

**Files:**
- Modify: `README.md`（如有必要补充"实验代码清理"一节）
- Create: `docs/validation/2026-08-08-refactor-experimental-code.md`

**Step 1: 全量回归**
Run:
```powershell
$env:PYTHONPATH="src"; python -m unittest discover -s tests/robot_edge -p "test_*.py" -q
$env:PYTHONPATH="src"; python -m unittest discover -s tests/cortex_navigation -p "test_*.py" -q
$env:PYTHONPATH="src"; python -m unittest tests.e2e.test_operator_console_mock -q
$env:PYTHONPATH="src"; python -m unittest tests.e2e.test_operator_robot_edge_fixture -q
git diff --check
```

**Step 2: 记录报告**
- 写 `docs/validation/2026-08-08-refactor-experimental-code.md`，记录每个任务的 PASS/FAIL、改动文件、测试命令与输出摘要、无硬件声明。

**Step 3: 提交**
```powershell
git add README.md docs/validation
git commit -m "docs: record experimental-code refactor validation"
```

**Acceptance:** 软件套件全绿；报告记录完整；无硬件/SDK 路径被误标记为已验证。

---

## 停止条件

- 任一 refactor 导致 `ubrobot_manipulation` ROS 包行为/API 改变而测试未同步通过；
- 任何改动让非 legacy 主路径（`cortex`/`cortex-mock`/`robot-edge` backend）依赖硬件或 SDK；
- `motion_arbitration.py` 或 `AuthorityTracker` 语义（lease/stationary 判定阈值）被改动；
- 遗漏 CI/工作站测试路径（`PYTHONPATH` 仅含 `src` 时无法通过）。
