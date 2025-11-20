# -*- coding: utf-8 -*-
"""
把 Copilot 导出的 chat.json 中的“助手回复”提取出来，
按 UTF-8 编码写入一个 txt 文件。

用法：
    python chat_to_txt.py chat.json              # 输出到 chat.txt
    python chat_to_txt.py chat.json out.txt     # 输出到 out.txt
"""

import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_assistant_replies(data):
    """
    从 JSON 结构里遍历出助手的回复文本。

    约定：顶层 data["requests"][i]["response"] 是一个列表，
    其中：
      - 有些元素是 {"kind": "..."} 之类的状态消息；
      - 真正的回复是那种“有 value 字段、但没有 kind 字段”的对象。
    我们只取这种对象的 value。
    """
    requests = data.get("requests", [])
    for i, req in enumerate(requests, 1):
        for item in req.get("response", []):
            if not isinstance(item, dict):
                continue
            # 过滤掉 toolInvocation / mcpServersStarting / terminal 等
            if "value" in item and "kind" not in item:
                text = str(item["value"]).replace("\r\n", "\n")
                yield i, text


def main():
    if len(sys.argv) < 2:
        print("用法: python chat_to_txt.py chat.json [output.txt]", file=sys.stderr)
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.is_file():
        print(f"找不到输入文件: {in_path}", file=sys.stderr)
        sys.exit(1)

    # 输出文件：未指定时，用同名的 .txt
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = in_path.with_suffix(".txt")

    data = load_json(in_path)

    # 直接按 UTF-8 写文件
    with out_path.open("w", encoding="utf-8") as out:
        for req_idx, reply in iter_assistant_replies(data):
            out.write(f"=== 对话 {req_idx} 的回复 ===\n")
            out.write(reply)
            if not reply.endswith("\n"):
                out.write("\n")
            out.write("\n")

    # 这里只输出一点点中文到 stderr（GBK 能正常表示）
    print(f"已写入 UTF-8 文件: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
