import re
import json
from collections import Counter
from typing import List

# ── Response 파싱 ──

def parse_toolcall_json(matches):
    results = []
    for m in matches:
        try:
            tool_call = json.loads(m)
            results.append(tool_call)
        except json.JSONDecodeError:
            continue
    return results

def parse_toolcall(response):
    """<tool_call> 블록에서 {name, arguments} dict를 추출하여 list로 반환. 실패 시 None."""
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    if not matches:
        return None
    return matches
    
def parse_action(response):
    """'Action: ...' 줄에서 자연어 설명을 추출. 실패 시 None."""
    match = re.search(r"Action:\s*(.+)", response)
    if not match:
        return None
    return match.group(1).strip()

def parse_thought(response):
    """thought를 '<think>' 없이 '...</think>' 형식으로 추출. 실패 시 None.
    '<think>...</think>' 또는 '...\n</think>'(여는 태그 없음) 모두 처리."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 여는 태그 없이 </think>로 끝나는 경우: </think> 앞의 내용을 thought로 추출
    match = re.search(r"^(.*?)\s*</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_thought_and_action(response):
    """<think>/Action: 태그 없이, 'I'll'/'I will'/'Let me' 등의 의도 표현을 기준으로
    thought(추론 부분)와 action(행동 설명 부분)을 분리.
    tool_call 블록은 제거한 뒤 텍스트만 대상으로 한다."""
    # tool_call 블록 제거
    text = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL).strip()
    if not text:
        return None, None

    # "I'll" / "I will" / "Let me" 등으로 시작하는 문장을 action 시작점으로 탐색
    # 문장 경계(. 또는 줄바꿈 뒤)에서 시작하는 패턴을 우선 매칭
    pattern = r"""(?:(?<=\.)\s*|(?<=\n)\s*|^)"""  # 문장 경계
    pattern += r"""((?:I'll|I will|Let me|Now I(?:'ll| will| need to| should| am going to| want to))\s.+)$"""
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        action = match.group(1).strip()
        thought = text[:match.start(1)].strip().rstrip(".")
        if thought:
            thought = thought.strip()
        else:
            thought = None
        return thought, action

    # 패턴 매칭 실패 시 전체를 thought로
    return text, None

def parse_response(response):
    """response 문자열에서 thought와 action을 파싱하여 dict로 반환."""
    thought = parse_thought(response)
    action = parse_action(response)
    tool = parse_toolcall(response)
    if tool is not None and thought is None and action is None:
        # re parse
        thought, action = parse_thought_and_action(response)

    return {
        "response": response,
        "thought": thought, 
        "action": action,
        "tool": tool,
    }

