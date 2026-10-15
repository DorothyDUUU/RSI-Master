"""常量和默认配置。"""

from __future__ import annotations

DEFAULT_PORT = 40059
DEFAULT_TIMEOUT = 300.0
MAX_OUTPUT_LENGTH = 100_000  # 字符数
MAX_CONCURRENT_PROCESSES = 60  # 最大并发进程数

# long_sleep 过滤器: 拦截 sleep 超过该秒数(默认 40 分钟)的命令
DEFAULT_MAX_SLEEP_SECONDS = 2400  # 40 * 60

# 默认白名单路径（可在启动时通过 --whitelist 追加部署侧特有路径）
DEFAULT_WHITELIST: list[str] = [
    "/root",
    "/tmp",
    "/var/tmp",
]

# 常见会带路径参数的命令
PATH_COMMANDS: set[str] = {
    "cd", "ls", "find", "grep", "cat", "head", "tail", "wc",
    "cp", "mv", "rm", "mkdir", "rmdir", "touch", "chmod", "chown",
    "ln", "readlink", "realpath", "stat", "file", "du", "df",
    "tar", "zip", "unzip", "gzip", "gunzip",
    "python", "python3", "pip", "pip3",
    "bash", "sh", "source", ".",
    "docker", "docker-compose",
    "git",
    "sed", "awk", "sort", "uniq", "diff", "comm",
    "tee", "xargs",
    "vim", "nano", "emacs",
    "less", "more",
    "tree",
}
