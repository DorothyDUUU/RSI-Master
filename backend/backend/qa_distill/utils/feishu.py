"""飞书机器人告警 (迁移自 thirdparty/agent-code-base codebase/_utils/_send_feishu_message.py)。

用于长批量蒸馏任务完成/失败时的飞书 Webhook 通知 (可选依赖)。
配置节 ``feishu.url`` / ``feishu.keyword``, 或直接传 webhook_url。
"""

from __future__ import annotations

from typing import Optional

import requests

from .config import get_config
from .logging import get_logger


class FeishuBot:
    """飞书 Webhook 机器人客户端, 封装消息发送与告警。"""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        keyword: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """初始化; webhook_url/keyword 缺省时从 config 的 feishu 节读取。"""
        self.logger = get_logger(__name__)

        if webhook_url is None or keyword is None:
            config = get_config(config_path)

            if hasattr(config, "feishu"):
                webhook_url = webhook_url or config.feishu.url
                keyword = keyword or config.feishu.keyword
            else:
                webhook_url = webhook_url or config.get("feishu.url")
                keyword = keyword or config.get("feishu.keyword")

        if not webhook_url:
            raise ValueError(
                "Feishu webhook URL not provided. "
                "Please provide webhook_url parameter or configure it in config.yaml "
                "under 'feishu.url'"
            )

        self.webhook_url = webhook_url
        self.keyword = keyword
        self.logger.info(f"FeishuBot initialized with webhook URL: {webhook_url[:30]}...")

    def send_message(self, message: str, add_keyword: bool = True, timeout: int = 10) -> bool:
        """发送文本消息; 返回是否成功。"""
        if add_keyword and self.keyword and self.keyword not in message:
            formatted_message = f"【{self.keyword}】{message}"
        else:
            formatted_message = message

        headers = {"Content-Type": "application/json"}
        payload = {"msg_type": "text", "content": {"text": formatted_message}}

        self.logger.info(f"Sending message to Feishu: '{message[:100]}...'")

        try:
            response = requests.post(
                self.webhook_url, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()

            response_data = response.json()
            if response_data.get("code") == 0 or response_data.get("StatusCode") == 0:
                self.logger.info("Message sent to Feishu successfully")
                return True
            else:
                self.logger.error(
                    f"Failed to send message to Feishu. Response: {response_data}"
                )
                return False

        except requests.exceptions.Timeout:
            self.logger.error(
                f"Timeout while sending message to Feishu (timeout={timeout}s)"
            )
            return False

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Network error occurred while sending message to Feishu: {e}"
            )
            return False

    def send_alert(self, title: str, content: str, add_keyword: bool = True) -> bool:
        """发送告警 (标题 + 详情)。"""
        message = f"⚠️ {title}\n\n{content}"
        return self.send_message(message, add_keyword=add_keyword)


# 全局单例 (兼容旧接口)
_global_bot: Optional[FeishuBot] = None


def get_feishu_bot(config_path: Optional[str] = None) -> FeishuBot:
    """获取全局 FeishuBot 单例。"""
    global _global_bot

    if _global_bot is None:
        _global_bot = FeishuBot(config_path=config_path)

    return _global_bot


def send_feishu_message(
    message: str, add_keyword: bool = True, config_path: Optional[str] = None
) -> bool:
    """便捷函数: 直接发一条飞书文本消息。"""
    bot = get_feishu_bot(config_path)
    return bot.send_message(message, add_keyword=add_keyword)
