"""설정 파일 로더"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """YAML 설정 파일 로더"""

    def __init__(self, config_dir: str = None):
        """
        Args:
            config_dir: 설정 파일 디렉토리 경로
        """
        if config_dir is None:
            # 프로젝트 루트/config 디렉토리
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / 'config'

        self.config_dir = Path(config_dir)

    def load(self, filename: str) -> Dict[str, Any]:
        """
        YAML 파일 로드

        Args:
            filename: 설정 파일명

        Returns:
            설정 딕셔너리
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config or {}

    def load_config(self) -> Dict[str, Any]:
        """기본 설정 파일 로드 (config.yaml)"""
        return self.load('config.yaml')

    def load_api_keys(self) -> Dict[str, Any]:
        """API 키 파일 로드 (api_keys.yaml)"""
        return self.load('api_keys.yaml')

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        점(.) 구분자를 사용한 중첩 키 접근

        Args:
            key_path: 키 경로 (예: "exchange.testnet")
            default: 기본값

        Returns:
            설정 값
        """
        config = self.load_config()
        keys = key_path.split('.')

        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value
