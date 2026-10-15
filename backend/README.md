# GPU Backend Design

## 宗旨

Agent 需要实时频繁的调用 vllm, llama-factory 等后端服务，Agent 部署在 CPU 机器上，对应的服务运行在 GPU 机器上，之前需要存在交互通信的格式

因此，提出 gpu_backend 这样一个 CLI 工具，agent 可以在 CPU 侧运行 client 对应的代码，对应代码通过 SSH 转接通信，将信号转接到远程的 GPU 机器上，实现后端训练服务的解耦

## Interface

### `client`

#### `Shell Command Interation`

关键在于执行 Shell 脚本，但是每一个 Shell 脚本都持久化一个唯一的 session_id，保存环境变量等环境

- `list_shell_session()`:
    - 无需传入任何参数
    - 展示当前激活的 shell 环境
    - 需要列出当前正在前台运行的 shell & 后台运行的 shell (见 `submit_shell_scripts`)
- `update_shell_session(command: str, timeout: int, session_id: str)`:
    - 在当前的 shell-session 中执行一条命令
    - 默认单次，前台执行
    - 返回 shell 的返回码，stdout，stderr 等信息
    - 如果不传入 session_id，默认启动一个新的 Shell:
        - 额外返回 pwd, shell_session_id 等元数据
- `kill_shell_session(session_id: str)`:
    - 杀死一个 shell session 及其内部创建的对应子进程
- `submit_shell_scripts(timeout: int, command_path: str, resume_session_id: str, run_in_background)`
    - `timeout (seconds)`: 超时上限
    - `command_path`: 写成 bash 脚本的形式
    - `session_id`: 是否会持久化(决定走脚本是 bash xx.sh 还是 source xx.sh)，如果填入就是在对应持久化的 shell 上继续运行
    - `run_in_background`: 是否运行在后台，如果运行在后台，则需要将日志重定向到某个特定的 log 文件，将 log 文件的路径返回到工具输出中
    * 后端服务你看看能不能通过 tmux 实现，同时重定向到 log 文件，比如后端服务就是启动一个 gpu_backend_<hash> 这样子的 tmux，然后如果进程结束了这个 tmux 就没了

#### Resource Allocation

可以远程监控 GPU 的CPU, 内存占用和显卡占用等

- `get_info()`
    - 列举出对应的信息

### `server`

Server 启动一个后端端口，前端的 server 解析用户 CLI 的命令行调用参数，然后调用后端的端口，后端端口进行执行，给前端信息

额外的 feature:

- 运行时长超过 5 分钟，自动转化为后台任务, 避免过长的工具气泡时间
- 添加并发资源总控

## File Structure

`backend/backend` 是给 Agent 看的一些参考脚本，你不需要动这里的文件

你需要在 backend 这里实现:

- server 的后端服务和对应的启动脚本
- client 的前端 CLI 封装
- 对应的测试脚本