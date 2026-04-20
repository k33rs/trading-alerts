from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from trading_bot.config import TelegramConfig


@dataclass(slots=True)
class TelegramNotifier:
    config: TelegramConfig

    def is_enabled(self) -> bool:
        return bool(self.config.enabled and self.config.bot_token and self.config.chat_id)

    def send(self, text: str) -> None:
        if not self.is_enabled():
            return
        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": self.config.chat_id, "text": text},
            timeout=20,
        )
        response.raise_for_status()

    def send_photo(self, photo_path: Path, caption: str | None = None) -> None:
        if not self.is_enabled():
            return
        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendPhoto"
        with photo_path.open("rb") as photo_file:
            response = requests.post(
                url,
                data={"chat_id": self.config.chat_id, "caption": caption or ""},
                files={"photo": photo_file},
                timeout=30,
            )
        response.raise_for_status()

    def send_signal(self, text: str, chart_path: Path | None = None, chart_caption: str | None = None) -> None:
        self.send(text)
        if chart_path is not None:
            self.send_photo(chart_path, chart_caption)