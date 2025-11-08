"""BingX API 예외 클래스"""


class BingXAPIError(Exception):
    """BingX API 기본 예외"""
    def __init__(self, message: str, code: int = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BingXAuthError(BingXAPIError):
    """인증 관련 예외"""
    pass


class BingXNetworkError(BingXAPIError):
    """네트워크 연결 예외"""
    pass


class BingXRateLimitError(BingXAPIError):
    """API 요청 제한 초과 예외"""
    pass


class BingXOrderError(BingXAPIError):
    """주문 관련 예외"""
    pass


class BingXInsufficientBalanceError(BingXAPIError):
    """잔고 부족 예외"""
    pass
