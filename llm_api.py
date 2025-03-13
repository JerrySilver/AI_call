import re
from openai import OpenAI

from logger_config import setup_logging, CustomAdapter

# 初始化日志
logger_base = setup_logging("api.log")
logger = CustomAdapter(logger_base, {})

# 设置你的OpenAI API密钥
OPENAI_API_URL = "https://api.siliconflow.cn/v1/"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

client = OpenAI(
    base_url=OPENAI_API_URL,
    api_key='',
)

# 预编译正则表达式：用于去除 Markdown 中的标题、列表、加粗等语法
TITLE_PATTERN = re.compile(r'^#{1,6}\s+', flags=re.MULTILINE)
ORDERED_LIST_PATTERN = re.compile(r'^\s*\d+\.\s+', flags=re.MULTILINE)
UNORDERED_LIST_PATTERN = re.compile(r'^\s*[-+*]\s+', flags=re.MULTILINE)


def markdown_to_plain_text(md_text: str) -> str:
    # 去除 Markdown 标题
    text = TITLE_PATTERN.sub('', md_text)
    # 去除有序列表标记
    text = ORDERED_LIST_PATTERN.sub('', text)
    # 去除无序列表标记
    text = UNORDERED_LIST_PATTERN.sub('', text)
    # 去除加粗、斜体和代码块标记
    text = text.replace('**', '').replace('*', '').replace('```', '')
    return text.strip()


def call_ai(prompt: str) -> str:
    messages = [
        {
            'role': 'user',
            'content': f'''{prompt}''',
        },
        # {
        #     'role': 'system',
        #     'content': "以文本格式输出内容,不要其他任意格式，例如html，markdown",
        # }
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        # 获取返回的 Markdown 格式文本
        result = response.choices[0].message.content.strip()
        # 使用预编译正则表达式将 Markdown 转换为普通文本
        # plain_text = markdown_to_plain_text(result)
        # return plain_text
        return result
    except Exception as e:
        logger.error("调用大模型接口出错", exc_info=True)
        return "调用大模型api出错: {}".format(e)


if __name__ == '__main__':
    print(call_ai(
        "请帮我解决一个算法问题:有一个从一楼到二楼的楼梯，楼梯的梯数有20个，小明一次可以爬一个楼梯也可以爬两个楼梯，请问小明到二楼有多少种走法"))
