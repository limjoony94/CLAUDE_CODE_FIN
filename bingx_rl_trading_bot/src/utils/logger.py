"""로깅 설정"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = None,
    rotation: str = "100 MB",
    retention: str = "30 days"
) -> None:
    """
    로거 설정

    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        log_file: 로그 파일 경로
        rotation: 로그 파일 로테이션 크기
        retention: 로그 파일 보관 기간
    """
    # 기존 핸들러 제거
    logger.remove()

    # 콘솔 출력 설정
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # 파일 출력 설정 (옵션)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

    logger.info(f"Logger initialized with level: {log_level}")
