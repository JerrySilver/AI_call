import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from ai_request import AIRequest
from logger_config import setup_logging, CustomAdapter
from llm_api import call_ai
from slowapi.errors import RateLimitExceeded
from fastapi.concurrency import run_in_threadpool

# 初始化日志
logger_base = setup_logging("api.log")
logger = CustomAdapter(logger_base, {})
# 禁用 slowapi 的日志传播，让它不向上传递日志记录
slowapi_logger = logging.getLogger("slowapi")
slowapi_logger.propagate = False

# 初始化限流器，每个 IP 每分钟最多调用 5 次
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    logger.info(f"该地址超过默认访问频率", extra={"ip": request.client.host})
    return JSONResponse(
        status_code=500,  # 返回状态码原本是429，但需求是所有异常返回500
        content={"code": 500, "message": "访问频率超过限制，每分钟最多允许5次调用"}
    )


@app.post("/ai_call")
@limiter.limit("5/minute")
async def ai_call_endpoint(request: Request, payload: AIRequest):
    prompt = payload.prompt
    if not prompt:
        response = {"code": 500, "message": "缺少提示词参数"}
        logger.info("请求参数缺失", extra={"ip": request.client.host})
        return JSONResponse(status_code=500, content=response)
    try:
        # 异步调用阻塞函数
        result = await run_in_threadpool(call_ai, prompt)
        response = {"code": 200, "message": result}
    except Exception as e:
        response = {"code": 500, "message": str(e)}
    logger.info(f"请求参数: {prompt} | 响应结果: {response}", extra={"ip": request.client.host})
    return JSONResponse(status_code=200 if response["code"] == 200 else 500, content=response)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
