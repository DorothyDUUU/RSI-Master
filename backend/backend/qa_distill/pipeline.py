"""并发 QA 数据蒸馏流水线 (迁移自 thirdparty/agent-code-base codebase/llm_pipeline/pipeline.py)。

用法一句话: 配好 config → 写好 system/user prompt 模板 → 写好输出解析器 →
``pipeline.run(data_pool, concurrency_limit, extract_function)`` 并发生成 JSONL。

迁移时相对上游做的修正 (见 README「迁移说明」):
    - 全部并发 worker 共享一个 LLMClient, 从而共享同一个 RateLimiter,
      使 ``rate_limit`` 真正按全局生效 (上游每任务新建 client, 限速形同虚设)。
    - 修复 ``safe_chat_completion`` 返回 None 时解包崩溃的 bug, 落一条带
      error 字段的结果而不是掩盖真实错误。
    - 配置文件里的相对路径 (prompt 文件 / output_dir) 一律按 config 文件所在
      目录解析, 不再依赖 CWD。
    - prompt 文件内容按路径缓存, 大批量时避免重复磁盘 IO。
    - 新增 ``run(..., resume=True)`` 断点续跑: 跳过 result.jsonl 里已完成条目。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm

from .llm_client import LLMClient, ModelConfig
from .utils.config import get_config
from .utils.logging import get_logger


class LLMPipeline:
    """自动化数据蒸馏流水线: 并发调用 LLM、结果提取、进度跟踪、JSONL 落盘。"""

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Args:
            config_path: 配置文件路径; None 时走默认查找 (QA_DISTILL_CONFIG env
                或 <包根>/config/config.yaml)。
            **kwargs: 点号路径 → 值的覆盖项 (如 ``model="gpt-4o"`` 或
                ``rate_limit=10``), 优先级高于 config 文件。
        """
        self.logger = get_logger(__name__)
        self.config = get_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # 配置文件所在目录: config 里的相对路径都相对它解析
        self._config_dir = None
        if getattr(self.config, "path", None) is not None:
            self._config_dir = self.config.path.parent

        # kwargs 覆盖 config
        if kwargs:
            self._update_config_with_kwargs(kwargs)

        # 输出目录 (含 experiment_dir / result.jsonl)
        self.output_dir = self._setup_output_dir()

        # prompt 文件路径
        self._load_prompt_paths()

        # 文件锁 + 结果缓存 + prompt 文件缓存
        self.file_lock = threading.Lock()
        self.results: List[Dict[str, Any]] = []
        self._prompt_cache: Dict[str, str] = {}

        # 供直接调用 run_single_queries 时使用; 正常 run() 流程会显式传入 client
        self.client: Optional[LLMClient] = None

        self.logger.info(f"LLMPipeline initialized: output_dir={self.output_dir}")

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #

    def _update_config_with_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """记录 kwargs 覆盖项 (Config 不可变, 单独存放覆盖表)。"""
        if not hasattr(self, "_config_overrides"):
            self._config_overrides: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if value is not None:
                self.logger.info(f"Overriding config: {key}={value}")
                self._config_overrides[key] = value

    def _get_config_value(self, key_path: str, default: Any = None) -> Any:
        """读取配置 (dot-notation), kwargs 覆盖优先。"""
        if hasattr(self, "_config_overrides"):
            if key_path in self._config_overrides:
                return self._config_overrides[key_path]
            short_key = key_path.replace("qa_distill.", "")
            if short_key in self._config_overrides:
                return self._config_overrides[short_key]
        return self.config.get(key_path, default)

    def _resolve_path(self, p: str) -> str:
        """
        把配置里的路径解析为绝对路径。

        规则: 绝对路径直接用; 相对路径优先按 config 文件所在目录解析 (存在则
        用之), 否则回退到当前工作目录; 都不存在 (如待创建的 output 目录) 时
        按 config 目录解析。
        """
        path = Path(p).expanduser()
        if path.is_absolute():
            return os.path.normpath(str(path))

        if self._config_dir is not None:
            cfg_relative = self._config_dir / path
            if cfg_relative.exists():
                return os.path.normpath(str(cfg_relative))

        if (Path.cwd() / path).exists():
            return os.path.normpath(str(Path.cwd() / path))

        if self._config_dir is not None:
            return os.path.normpath(str(self._config_dir / path))
        return os.path.normpath(str(path))

    def _load_prompt_paths(self) -> None:
        """从配置读取 prompt 文件路径 (已解析为绝对路径)。"""
        self.system_prompt_path = self._resolve_path(
            self._get_config_value("qa_distill.prompts.system_prompt_path")
        )
        self.user_prompt_path = self._resolve_path(
            self._get_config_value("qa_distill.prompts.user_prompt_path")
        )

    def _setup_output_dir(self) -> str:
        """建立输出目录, 初始化 result.jsonl 路径。"""
        output_dir = self._get_config_value("qa_distill.output_data.output_dir", "output")
        experiment_name = self._get_config_value(
            "qa_distill.output_data.experiment_name", "experiment"
        )
        need_timestamp = self._get_config_value(
            "qa_distill.output_data.need_time_stamp", False
        )

        output_dir = self._resolve_path(output_dir)
        if need_timestamp:
            experiment_dir = os.path.join(output_dir, experiment_name, self.timestamp)
        else:
            experiment_dir = os.path.join(output_dir, experiment_name)

        self.experiment_dir = experiment_dir
        self.logger.info(f"Experiment directory: {experiment_dir}")

        self.experiment_path = os.path.join(experiment_dir, "result.jsonl")
        os.makedirs(experiment_dir, exist_ok=True)

        return experiment_dir

    def _get_model_config(self) -> ModelConfig:
        """
        从配置构建 ModelConfig。

        api_key / base_url 缺省时回退到 ``.env`` 或环境变量
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` (与仓库 llm.openai 约定一致)。
        """
        try:
            api_key = self._get_config_value("qa_distill.model.api_key")
            base_url = self._get_config_value("qa_distill.model.base_url")
            model_name = self._get_config_value(
                "qa_distill.model.model", "gpt-4o-mini"
            )
            max_tokens = self._get_config_value("qa_distill.model.max_tokens", 5012)
            temperature = self._get_config_value("qa_distill.model.temperature", 0.7)
            rate_limit = self._get_config_value("qa_distill.model.rate_limit", 20)

            # .env / 环境变量兜底: .env 只填空, 不覆盖已显式设置的环境变量
            if not api_key or not base_url:
                env_path = Path.cwd() / ".env"
                if env_path.exists():
                    from dotenv import load_dotenv

                    load_dotenv(env_path)  # override=False
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                base_url = base_url or os.getenv("OPENAI_BASE_URL")

            if not api_key or not base_url:
                raise ValueError("API key and base URL are required in configuration")

            return ModelConfig(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                rate_limit=rate_limit,
            )

        except Exception as e:
            raise ValueError(
                f"Failed to load model configuration: {e}. "
                "Please ensure config.yaml contains a 'qa_distill.model' section "
                "with api_key, base_url, and model fields (or set OPENAI_API_KEY / "
                "OPENAI_BASE_URL)."
            )

    # ------------------------------------------------------------------ #
    # Prompt 加载 / 渲染
    # ------------------------------------------------------------------ #

    def _load_prompt_from_file(self, file_path: str, **format_variables) -> str:
        """
        读取 prompt 文件并按 ``.format(**kwargs)`` 渲染占位符 (按路径缓存原文)。

        Args:
            file_path: prompt 文件绝对路径。
            **format_variables: ``{place_holder}`` 填充变量。

        Returns:
            渲染后的 prompt 字符串; 文件不存在或渲染失败时返回空串。
        """
        if not file_path:
            self.logger.error("Empty prompt file path")
            return ""

        if file_path not in self._prompt_cache:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    self._prompt_cache[file_path] = file.read().strip()
            else:
                self.logger.error(f"Failed to load prompt file: {file_path}")
                return ""

        prompt = self._prompt_cache[file_path]
        try:
            prompt = prompt.format(**format_variables)
        except KeyError as e:
            self.logger.warning(
                f"Prompt formatting error, missing key: {e}. Using original prompt."
            )
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {e}")
        return prompt

    # ------------------------------------------------------------------ #
    # 结果落盘
    # ------------------------------------------------------------------ #

    def save_result(self, result: Dict[str, Any]) -> None:
        """把一条结果追加写入 result.jsonl (线程/协程安全)。"""
        with self.file_lock:
            with open(self.experiment_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            self.results.append(result)

    # ------------------------------------------------------------------ #
    # 结果提取 (输出解析器)
    # ------------------------------------------------------------------ #

    def extract_all(self, pattern_name: str, text: str) -> List[str]:
        """
        提取文本中所有 ``<pattern_name>...</pattern_name>`` 标签内容。

        支持多行、属性、大小写不敏感。

        Example:
            >>> extract_all("draft", "<draft>hello</draft>")
            ["hello"]
        """
        if text is None:
            return []
        tag = re.escape(pattern_name.strip().lower())
        pattern = rf"<{tag}\b[^>]*>\s*(.*?)\s*</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches]

    def make_default_extractor(
        self, pattern_name: str
    ) -> Callable[[str, bool], Optional[Union[str, List[str]]]]:
        """
        为某个标签创建默认提取函数。

        返回的 extract(response, to_list=True): to_list=True 返回列表,
        否则返回第一个匹配; 未匹配返回 [] / None。
        """
        logger = self.logger

        def extract(response: str, to_list: bool = True) -> Union[List[str], str, None]:
            try:
                extract_list = self.extract_all(pattern_name, response)

                if not extract_list:
                    logger.warning(f"No <{pattern_name}> found in response.")
                    return [] if to_list else None

                return extract_list if to_list else extract_list[0]

            except Exception as e:
                logger.error(f"Extraction failed for <{pattern_name}>: {e}")
                return [] if to_list else None

        return extract

    # ------------------------------------------------------------------ #
    # 单条执行
    # ------------------------------------------------------------------ #

    async def run_single_queries(
        self,
        user_prompt: str,
        system_prompt: str,
        extract_function=None,
        input_data=None,
        client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        """
        单次 LLM 调用: 组织 system/user 消息 → 调用 → 提取 → 落盘。

        API 失败 (重试耗尽/超时) 时返回带 ``error`` 字段的结果, 而不是崩溃。
        """
        client_to_use: LLMClient = client if client is not None else self.client

        result = await client_to_use.safe_chat_completion(
            prompt=user_prompt, system_prompt=system_prompt
        )

        if result is None:
            error_result = {
                "input": input_data,
                "response": None,
                "error": "LLM call failed (retries exhausted or timeout)",
                "timestamp": datetime.now().isoformat(),
            }
            self.save_result(error_result)
            return error_result

        response, naive_response = result

        result_dict: Dict[str, Any] = {
            "input": input_data,
            "response": response,
            "naive_response": naive_response,
            "timestamp": datetime.now().isoformat(),
        }

        if extract_function:
            result_dict["extracted"] = extract_function(response)

        self.save_result(result_dict)
        return result_dict

    async def run_single_task(
        self,
        i: int,
        input_data: Dict[str, Any],
        extract_function: Optional[Callable] = None,
        client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        """
        处理单个 data_pool 条目: 渲染 system/user prompt → run_single_queries。

        input_data 可含:
            - ``system_prompt_kwargs``: 填充 system prompt 模板的变量
            - ``user_prompt_kwargs``: 填充 user prompt 模板的变量
            - ``input``: 任意元数据, 原样写进结果
        """
        try:
            self.logger.debug(f"Processing task {i + 1}")
            self.logger.debug(f"Input data: {input_data}")

            system_prompt_kwargs = input_data.get("system_prompt_kwargs", {})
            system_prompt = self._load_prompt_from_file(
                self.system_prompt_path, **system_prompt_kwargs
            )

            user_prompt_kwargs = input_data.get("user_prompt_kwargs", {})
            user_prompt = self._load_prompt_from_file(
                self.user_prompt_path, **user_prompt_kwargs
            )

            return await self.run_single_queries(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                input_data=input_data,
                extract_function=extract_function,
                client=client,
            )

        except Exception as e:
            self.logger.error(f"Error processing task {i + 1}: {e}")
            error_result = {
                "input": input_data,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self.save_result(error_result)
            return error_result

    # ------------------------------------------------------------------ #
    # 并发执行
    # ------------------------------------------------------------------ #

    async def run_all_tasks(
        self,
        data_pool: List[Dict[str, Any]],
        concurrency_limit: int = 5,
        extract_function: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        并发运行全部任务。

        所有 worker 共享同一个 LLMClient → 共享同一个 RateLimiter,
        ``rate_limit`` 对全局生效。
        """
        sem = asyncio.Semaphore(concurrency_limit)
        model_config = self._get_model_config()
        client = LLMClient(config=model_config)

        async def worker(i, input_data):
            async with sem:
                return await self.run_single_task(i, input_data, extract_function, client)

        tasks = [worker(i, x) for i, x in enumerate(data_pool)]
        results: List[Dict[str, Any]] = []

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await coro)

        return results

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def _load_existing_inputs(self) -> List[Any]:
        """读取 result.jsonl 里已完成的 input 列表 (用于断点续跑)。"""
        if not os.path.exists(self.experiment_path):
            return []

        existing = []
        with open(self.experiment_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "input" not in rec or rec.get("error"):
                    continue
                if rec.get("response") is None:
                    continue
                if "extracted" in rec and not rec.get("extracted"):
                    continue
                existing.append(rec["input"])
        return existing

    def run(
        self,
        data_pool: List[Dict[str, Any]],
        concurrency_limit: int = 5,
        extract_function: Optional[Callable] = None,
        resume: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        运行蒸馏流水线。

        Args:
            data_pool: 输入条目列表; 每条含 ``system_prompt_kwargs`` /
                ``user_prompt_kwargs`` (可选) 与 ``input`` 元数据 (可选)。
            concurrency_limit: 最大并发数 (默认 5)。
            extract_function: 输出解析器 ``fn(response) -> 结构化结果``;
                None 时只保存原始 response。
            resume: 为 True 时跳过 result.jsonl 中已完成 (input 相同) 的条目,
                实现断点续跑。

        Returns:
            全部处理结果列表。结果已实时写入 result.jsonl。
        """
        self.logger.info("Starting QA distillation pipeline")
        self.logger.info(f"Concurrency limit: {concurrency_limit}")
        self.logger.info(f"Total tasks: {len(data_pool)}")

        if resume:
            done = {
                json.dumps(inp, ensure_ascii=False, sort_keys=True)
                for inp in self._load_existing_inputs()
            }
            todo = [
                item
                for item in data_pool
                if json.dumps(item, ensure_ascii=False, sort_keys=True) not in done
            ]
            skipped = len(data_pool) - len(todo)
            if skipped:
                self.logger.info(f"Resume: skip {skipped} already-finished items")
            data_pool = todo

        if not data_pool:
            self.logger.info("Nothing to process, returning early")
            return []

        results = asyncio.run(
            self.run_all_tasks(
                data_pool=data_pool,
                concurrency_limit=concurrency_limit,
                extract_function=extract_function,
            )
        )

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {i + 1} failed with exception: {result}")
                processed_results.append(
                    {"error": f"Task failed with exception: {result}", "index": i}
                )
            else:
                processed_results.append(result)

        self.logger.info("QA distillation pipeline completed")
        return processed_results
