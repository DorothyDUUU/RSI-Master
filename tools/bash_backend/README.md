# Bash Backend Server

安全的 bash 命令执行 HTTP 服务，支持可插拔过滤链、输出检查和进程管理。

## 目录结构

```
bash_backend/
├── __init__.py          # 包导出
├── __main__.py          # 服务启动入口
├── client.py            # CLI 客户端
├── constants.py         # 常量和默认配置
├── filter_base.py       # 过滤器基类和注册系统
├── command_parser.py    # 命令解析和路径提取
├── executor.py          # 命令执行器
├── tracker.py           # 进程跟踪器
├── monitor.py           # 系统资源监控
├── models.py            # API Pydantic 模型
├── app.py               # FastAPI 应用和端点
├── test/                # pytest 测试
│   ├── conftest.py
│   ├── test_command_parser.py
│   ├── test_filter_base.py
│   ├── test_filters.py
│   ├── test_executor.py
│   ├── test_tracker.py
│   ├── test_monitor.py
│   └── test_app.py
└── filters/             # 内置过滤器
    ├── __init__.py
    ├── sandbox.py       # 文件系统沙箱
    ├── dangerous.py     # 危险命令拦截
    ├── network.py       # 网络访问控制
    └── output_length.py # 输出长度检查
```

## 启动服务

```bash
# 模块方式启动（推荐）
.venv/bin/python -m tools.bash_backend --port 40059 --workspace /your/workspace
```

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `40059` | 服务端口 |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--workspace` | 空 | 默认工作目录 |
| `--whitelist` | 无 | 额外白名单路径（追加到默认白名单） |
| `--memory-limit` | 无 | 内存限制，如 `4g`、`512m` |
| `--cpu-limit` | 无 | CPU 时间限制（秒） |
| `--cpu-devices` | 无 | CPU 核心绑定，如 `0-7` |
| `--max-output` | `100000` | 最大输出字符数 |

完整示例（严格沙箱——只放行 workspace + 显式白名单）：

```bash
.venv/bin/python -m tools.bash_backend \
  --port 40059 \
  --workspace /path/to/workspace \
  --whitelist /path/to/workspace \
             /path/to/shared/datasets \
             /path/to/models
```

> 启动 bash backend(:40059) 后 `export BASH_BACKEND_URL=http://localhost:40059`,
> agent 的 `execute_bash` 会自动走它。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/execute` | 执行 bash 命令 |
| `GET` | `/processes` | 列出进程（支持 `?status=running` 过滤） |
| `POST` | `/kill/{process_id}` | kill 指定进程 |
| `POST` | `/kill_all` | kill 所有活跃进程 |
| `GET` | `/filters` | 列出所有过滤器 |
| `POST` | `/filters/{name}/toggle` | 启用/禁用过滤器 |
| `GET` | `/status` | 系统 CPU/内存状态 |

### curl 示例

```bash
# 执行命令
curl -X POST http://localhost:40059/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "ls -la", "agent_workspace": "/your/workspace"}'

# 查看进程
curl http://localhost:40059/processes
curl http://localhost:40059/processes?status=running

# kill 进程
curl -X POST http://localhost:40059/kill/<process_id>
curl -X POST http://localhost:40059/kill_all

# 系统状态
curl http://localhost:40059/status
```

## CLI 客户端

提供命令行客户端，方便查看进程和管理服务。

```bash
# 查看所有进程
.venv/bin/python -m tools.bash_backend.client ps

# 只看运行中的
.venv/bin/python -m tools.bash_backend.client ps --status running

# 查看系统资源状态
.venv/bin/python -m tools.bash_backend.client status

# kill 特定进程
.venv/bin/python -m tools.bash_backend.client kill <process_id>

# kill 全部活跃进程（带确认）
.venv/bin/python -m tools.bash_backend.client kill-all

# 跳过确认直接 kill 全部
.venv/bin/python -m tools.bash_backend.client kill-all --yes

# 执行命令
.venv/bin/python -m tools.bash_backend.client exec "echo hello"
.venv/bin/python -m tools.bash_backend.client exec "ls -la" --workspace /tmp
.venv/bin/python -m tools.bash_backend.client exec "python train.py" --timeout 600

# 列看过滤器
.venv/bin/python -m tools.bash_backend.client filters

# 禁用过滤器
.venv/bin/python -m tools.bash_backend.client toggle filesystem_sandbox --off

# 启用过滤器
.venv/bin/python -m tools.bash_backend.client toggle filesystem_sandbox --on

# 健康检查
.venv/bin/python -m tools.bash_backend.client health

# 指定其他服务器地址
.venv/bin/python -m tools.bash_backend.client -s http://other-host:40058 ps
```

## 过滤器系统

### 内置过滤器

#### Pre-filter（执行前拦截）

| 名称 | 默认状态 | 说明 |
|------|----------|------|
| `filesystem_sandbox` | ✅ 启用 | 限制文件系统访问，只允许 workspace 和白名单路径（委托 `tools/bash_security.check_paths`） |
| `dangerous_command` | ✅ 启用 | 危险命令拦截（委托 `tools/bash_security.check_command`） |
| `network_access` | ❌ 禁用 | 网络访问控制（可手动开启扩展） |

#### Post-check（执行后检查）

| 名称 | 默认状态 | 说明 |
|------|----------|------|
| `output_length` | ✅ 启用 | 输出超过 100k 字符时拒绝返回，提示用 `head`/`tail` |

### 默认白名单（constants.DEFAULT_WHITELIST）

以下路径在沙箱过滤器中默认允许访问：

```
/root
/tmp
/var/tmp
```

可通过 `--whitelist` 启动参数追加部署侧特有路径（如数据集 / 模型 / 缓存目录）。

> **白名单以服务端为准**：`--whitelist` + `DEFAULT_WHITELIST` 是唯一来源,客户端请求里的
> whitelist 会被服务端覆盖,不能擅自缩小/扩大沙箱。
>
> **严格沙箱**：只做精确匹配 + 前向前缀匹配,不放行白名单目录的「祖先」路径——
> 白名单里 `/a/b/public` 不会连带放行 `/a/b` 或 `/`。

### 沙箱跳过的命令

- `python` / `python3` 开头的命令跳过路径检查（脚本内部文件访问无法静态分析）。
- `cat` 等直接读文件的命令**不**跳过——`cat /etc/passwd` 会被拦截。

## 安全拦截层（tools/bash_security）

`tools/bash_security/` 是**纯 stdlib 的共享拦截模块**（不依赖 fastapi/pydantic），
由 `dangerous_command` / `filesystem_sandbox` 两个过滤器和 evomaster BashTool 的
兜底路径共用同一份拦截逻辑。策略移植自 Claude Code 泄露源码
`thirdparty/claude_code_source/src/tools/BashTool/{bashSecurity.ts, pathValidation.ts}`。

```
tools/bash_security/
├── __init__.py       # 导出 check_command / check_paths / SecurityResult
├── result.py         # SecurityResult(allowed, reason, check_id)
├── patterns.py       # 正则常量(命令替换/混淆引号/Zsh/危险命令…)
├── shelllex.py       # 引号剥离 / heredoc 抽取 / 顶层分隔符扫描器
├── interceptor.py    # check_command: 命令形状层(递归分析 $()/`/进程替换)
└── pathguard.py      # check_paths: 路径边界层(重定向/操作数/允许目录)
```

两层拦截：

1. **`check_command`** — 命令形状层：控制字符 / Unicode 空白 / 反斜杠转义运算符 /
   Zsh `=cmd` 展开 / 进程替换 `<( )` `>( )` / `~[...]` 等危险展开 / 反引号命令替换 /
   ANSI-C `$'…'`、locale `$"…"` 混淆引号 / 引号内 shell 元字符 / 变量紧邻重定向 /
   `/proc/*/environ` / IFS 注入 / eval 类内建（`eval` `exec` `trap` `source` 等，带
   `command -v`、`fc -l`、`compgen -c` 同款豁免）/ jq `system()` / fork bomb、
   `dd of=/dev/`、`rm -rf /` 等危险系统命令。
2. **`check_paths`** — 路径边界层：只对 PATH_COMMANDS 命令集抽取路径操作数；处理
   `--` 结束选项、`-o/--output=` 值、grep/rg pattern 分割、mv/cp 任何 flag →
   reject、find `-exec/-delete`、cd 后写路径 fail-closed、危险删除路径、
   `~user`/`$VAR`/glob 写路径、UNC 路径；解析为 realpath 后校验落在
   `workspace + 白名单` 内。**递归分析 `$()` `${}` `$[]` 反引号内部命令**——
   `echo $(cat /etc/passwd)` 会被拦截,而 `echo $(date)` 放行。

返回的 `SecurityResult` 带 `check_id`,被过滤器包装成 `[{check_id}] {reason}` 放进
filtered 响应的 stderr,便于日志排查具体命中了哪条规则。

### 客户端兜底拦截（后端挂掉时）

`evomaster/agent/tools/builtin/bash.py` 在后端服务不可用、fallback 到裸
`session.exec_bash` 之前,先用 `tools.bash_security` 做同样的两层检查:

```python
_blocked_by_security(command, workspace)
  → check_command(command)          # 命令形状
  → check_paths(command, cwd=workspace, allowed_dirs=[workspace] + 白名单)
```

命中 → **不执行**,返回 `[BLOCKED by security] {reason}`(exit_code=-1),行为与后端
filtered 响应一致。兜底白名单 = `[workspace] + DEFAULT_WHITELIST +
$BASH_BACKEND_WHITELIST`,与启动服务端时传给 `--whitelist` 的值共用同一份真相,
避免后端挂掉时 agent 读部署侧放行的路径被误拦。

### 逃逸测试组件

```bash
.venv/bin/python -m pytest tools/bash_security/tests/ -q                 # 核心逃逸组件
.venv/bin/python -m pytest tools/bash_backend/test/ -q                   # 后端全量(含升级后过滤器)
.venv/bin/python -m pytest evomaster/agent/tools/test_bash_security_fallback.py -q  # 兜底集成
```

`tools/bash_security/tests/test_escape_behaviors.py` 表格化覆盖两层拦截的已知逃逸
payload(每个都对应一个 check_id),外加良性用例防误伤回归。

### 动态管理过滤器

```bash
# 通过 API
curl -X POST http://localhost:40058/filters/filesystem_sandbox/toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# 通过 CLI
.venv/bin/python -m tools.bash_backend.client toggle filesystem_sandbox --off
```

### 自定义过滤器

继承 `BashFilter` 并用 `@register_filter` 装饰即可添加新规则：

```python
from tools.bash_backend.filter_base import (
    BashFilter, FilterAction, FilterContext, FilterResult, register_filter
)

@register_filter
class MyCustomFilter(BashFilter):
    name = "my_custom_filter"
    description = "自定义过滤规则"
    is_pre_filter = True  # True = 执行前检查, False = 执行后检查

    def check(self, context: FilterContext, output: str = "") -> FilterResult:
        # 在此实现你的过滤逻辑
        if "forbidden" in context.command:
            return FilterResult(
                action=FilterAction.REJECT,
                reason="包含禁止的关键词",
                filter_name=self.name,
            )
        return FilterResult(action=FilterAction.ALLOW, filter_name=self.name)
```

将文件放入 `filters/` 目录并在 `filters/__init__.py` 中导入即可自动注册。

## 运行测试

```bash
.venv/bin/python -m pytest tools/bash_backend/test/ -v
```

## 执行流程

```
客户端请求 /execute
    │
    ▼
Pre-filter Chain ──→ 任一 REJECT ──→ 返回 filtered 响应
    │ ALLOW
    ▼
CommandExecutor.execute()
    │ (进程启动后立刻更新 tracker 的 pid/pgid)
    │ (等待命令完成或超时)
    ▼
Post-check Chain ──→ 任一 REJECT ──→ 返回截断响应
    │ ALLOW
    ▼
返回完整结果
```
