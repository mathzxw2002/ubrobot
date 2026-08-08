# 实验性代码重构与生产加固 Validation Report

- Date: 2026-08-08
- Branch: `go2-piper-cortex-integration`
- Machine role: workstation (Windows dev host); all hardware disconnected
- Hardware: **none connected** — Go2, LeKiwi, Piper, SO101 all absent
- Hardware/SDK/cloud paths: **not executed** (no devices, no vendor SDKs)

## Objective

在无硬件空窗期把实验性代码整理到可维护状态:修复跨边界依赖、隔离 legacy
硬件直连路径、清理死代码、硬化启动脚本,并完成生产加固(P0 工程基础设施、
P1 代码健壮性、P2 可观测性/部署、P3 测试纵深)。所有改动的验收门槛是软件
测试全绿且无硬件路径被误标为已验证。

## Scope (this round)

| 计划 | Task | 内容 |
|---|---|---|
| 2026-08-08-refactor-experimental-code.md | Task 1 | `AuthorityTracker` 平移至 `ubrobot_contracts.motion_authority`;ROS 侧 re-export |
| 同上 | Task 2 | 隔离 `Go2Manager`/`UnitreeGo2Robot` legacy 硬件直连 |
| 同上 | Task 3 | 硬化 `start_console_hardware.ps1`(token 新鲜度/-ForceRefresh/工具校验/安全提示) |
| 同上 | Task 4 | 归档死代码 `arm_action.py`;README 状态表;legacy-rollback 文档 |
| 2026-08-08-production-hardening.md | P0 | CI 流水线(4 job)、ruff 配置、gitleaks + 静态秘密扫描 |
| 同上 | P1 | 结构化日志、异常纪律、mypy(46 文件)、pydantic-settings 集中配置 |
| 同上 | P2 | Prometheus `/v1/metrics`、非 root 容器、semver/cosign 发布文档 |
| 同上 | P3 | 覆盖率门禁(86%)、故障注入测试、hardware-contract CI |

## Software test results (all green)

| Suite | Count | Result |
|---|---|---|
| robot_edge (incl. motion authority, settings, metrics, fault injection) | 252 | OK |
| cortex_navigation | 202 | OK (Linux; Windows flaky = uvicorn `resource.getpagesize`, env-only) |
| e2e (console mock + robot-edge fixture) | 11 | OK |
| legacy rollback imports | 4 | OK |
| security (hardcoded secrets) | 4 | OK |
| coverage (core pure-Python, --fail-under=80) | — | 86% PASS |

## Static analysis

| Tool | Scope | Result |
|---|---|---|
| ruff check | production packages | clean (0 violations) |
| ruff format --check | production packages | clean |
| mypy | contracts + robot_edge + chat_ui | 46 files, no issues |
| gitleaks (working tree) | whole repo | no leaks (allowlisted vendored/tests/local creds) |
| coverage | core (excl. ROS backend) | 86% |

## Task 3 acceptance (start_console_hardware.ps1)

`scripts/hardware/test_start_console_hardware_params.ps1` — 10 checks PASS:

- required-tool (ssh/scp) guard present
- token freshness window + `LastWriteTime` check
- `-ForceRefresh` support
- hardware-authority safety notice
- state commands (status/logs/stop) delegate without token fetch
- missing cache triggers fetch; fresh cache reuses without fetch; `-ForceRefresh`
  re-fetches despite fresh cache

## Key decisions / notes

- **ROS 后端排除在 coverage 门槛外**:rclpy/相机话题无法在工作站执行,计入
  会导致门槛永久失败且无测量意义;由 robot 镜像 + 真机验收覆盖。
- **git 历史清理仍为 owner 决策**:已泄露的 DashScope key 在历史中
  (`ec3ee21`, 早期 startup 脚本),CI 扫工作树保护未来;filter-repo 重写 +
  force push 待 owner 批准。
- **Windows 偶发失败**:`cortex_navigation` 在 Windows 上偶发
  `resource.getpagesize`(uvicorn 兼容),Linux CI runner 通过,非代码回归。
- **集中配置保留的 env 读取**:`UBROBOT_SHUTDOWN_TOKEN`(运行时进程机密)与
  `RobotEdgeBackend` token 兜底(显式参数优先)。

## Hardware statement

No hardware acceptance was performed. All fixture/mock successes are
software-only and do NOT constitute evidence of robot capability. Any real
hardware stage requires the documented gates (physical E-stop, operator,
`--hardware` driver, staged plan).
