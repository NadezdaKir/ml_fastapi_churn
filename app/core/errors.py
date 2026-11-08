import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("churn_service")


class ChurnServiceError(Exception):
    def __init__(self, status_code, code, message, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_response(status_code, code, message, details=None):
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message, "details": details},
    )


def register_error_handlers(app: FastAPI):
    @app.exception_handler(ChurnServiceError)
    async def churn_error_handler(request: Request, exc: ChurnServiceError):
        logger.warning(
            "%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message
        )
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(
            "%s %s -> http_error: %s", request.method, request.url.path, exc.detail
        )
        return error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        logger.warning("%s %s -> validation_error", request.method, request.url.path)
        return error_response(
            422,
            "validation_error",
            "request data does not match the expected schema",
            jsonable_encoder(exc.errors()),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception("%s %s -> unexpected error", request.method, request.url.path)
        return error_response(500, "internal_error", "unexpected server error")