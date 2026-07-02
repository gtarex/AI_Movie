#!/usr/bin/env python3
"""DeepSeek Novel Writer - Web UI (Gradio)"""

import os, re, json
import socket
import gradio as gr
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from config import STORIES_ROOT, STORY_DIR as DEFAULT_STORY_DIR, MAX_HISTORY_PAIRS, MAX_PLANNING_TURNS, MAX_TOKENS_DEFAULT, MAX_TOKENS_WORLD, MAX_TOKENS_CHARACTER_PROFILE, MAX_TOKENS_SCENE_SETUP, MAX_TOKENS_MEMORY_COMPRESSION, MAX_TOKENS_SETTLEMENT, MAX_TOKENS_SCENE_MEMORY, MAX_TOKENS_FACT_PACK, MAX_TOKENS_FACT_CHECK, MAX_TOKENS_PLANNING_PROPOSAL, MAX_TOKENS_PLANNING_CHAT, MAX_TOKENS_STORY_EVENT, MAX_TOKENS_SCENE_SYNOPSIS, DEFAULT_REPLY_CHARS, TOKEN_MULTIPLIER_PER_CHAR, compute_max_tokens

# Monkey-patch Gradio 4.44.1 bug: additionalProperties:true crashes schema parser
import gradio_client.utils as grc_utils
_orig_get_type = grc_utils.get_type
def _patched_get_type(schema):
    if schema is True or schema is False or not isinstance(schema, dict):
        return "string"
    return _orig_get_type(schema)
grc_utils.get_type = _patched_get_type

DEFAULT_MODEL = "deepseek-v4-pro"
MODEL_CHOICES = ["deepseek-v4-pro", "deepseek-v4-flash"]
DEFAULT_THINKING_TYPE = "enabled"
THINKING_TYPE_CHOICES = ["enabled", "disabled"]
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_CHOICES = ["high", "max"]
RECENT_CONTEXT_MESSAGES = 10
UNDO_STACK_LIMIT = 10
MAX_FACT_REWRITE_ATTEMPTS = 2
STORY_HISTORY_CONTEXT_CHARS = 20000

_global_client = None
_global_api_key = None
_global_thinking_type = DEFAULT_THINKING_TYPE
_global_reasoning_effort = DEFAULT_REASONING_EFFORT

def read_file(fp):
    if not os.path.exists(fp): return ""
    with open(fp,"r",encoding="utf-8") as f: return f.read().strip()

def read_file_raw(fp):
    if not os.path.exists(fp): return ""
    with open(fp,"r",encoding="utf-8") as f: return f.read()

def write_file(fp, c):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp,"w",encoding="utf-8") as f: f.write(c.strip())

def write_file_raw(fp, c):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp,"w",encoding="utf-8") as f: f.write(c)

def append_file(fp, c):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp,"a",encoding="utf-8") as f: f.write("\n\n"+c.strip())

def outline_path(sdir):
    return os.path.join(sdir, "outline.txt")

def normalize_story_dir(sdir):
    raw = (sdir or DEFAULT_STORY_DIR).strip()
    if not raw:
        return DEFAULT_STORY_DIR
    norm = os.path.normpath(raw)
    if os.path.isabs(norm):
        return norm
    root = os.path.normpath(STORIES_ROOT)
    if norm in (".", root):
        return DEFAULT_STORY_DIR
    if norm.startswith(root + os.sep):
        return norm
    if os.sep not in norm:
        return os.path.join(root, norm)
    return norm

def current_story_dir(state):
    if not isinstance(state, dict):
        return DEFAULT_STORY_DIR
    sdir = normalize_story_dir(state.get("story_dir", DEFAULT_STORY_DIR))
    state["story_dir"] = sdir
    return sdir

def list_dirs(base=None):
    root = os.path.normpath(base or STORIES_ROOT)
    if root in (".", ""):
        root = os.path.normpath(STORIES_ROOT)
    if not os.path.isdir(root): return [DEFAULT_STORY_DIR]
    ds = [os.path.join(root,d) for d in os.listdir(root) if os.path.isdir(os.path.join(root,d)) and not d.startswith(".")]
    return sorted(ds) if ds else [DEFAULT_STORY_DIR]

def get_char_dir(sdir):
    return os.path.join(sdir, "char")

def ensure_char_dir(sdir):
    cdir = get_char_dir(sdir)
    os.makedirs(cdir, exist_ok=True)
    return cdir

def character_path(sdir, codename):
    return os.path.join(get_char_dir(sdir), f"{codename}.txt")

def get_current_scene_file(sdir):
    for n in range(1,1000):
        fn = f"scene_{n:02d}.txt"; fp = os.path.join(sdir,fn)
        if not os.path.exists(fp): return fp, fn, n
    return os.path.join(sdir,"scene_999.txt"), "scene_999.txt", 999

def scene_number_from_name(scene_name):
    m = re.match(r"scene_(\d+)\.txt$", scene_name or "")
    return int(m.group(1)) if m else 1

def list_scene_files(sdir):
    if not os.path.isdir(sdir):
        return []
    scenes = []
    for fn in os.listdir(sdir):
        if re.match(r"scene_\d+\.txt$", fn or ""):
            scenes.append(fn)
    return sorted(scenes, key=lambda name: scene_number_from_name(name))

def find_free_port(start_port=7860, tries=50):
    for port in range(start_port, start_port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port

def build_turn_order(active_characters):
    return [c for c in active_characters if c]

def get_current_actor(state):
    order = state.get("turn_order",[])
    if not order: return None
    return order[state.get("turn_index",0) % len(order)]

def advance_turn(state):
    order = state.get("turn_order",[])
    state["turn_index"] = (state.get("turn_index",0) + 1) % len(order) if order else 0
    return state

def turn_label(state):
    return get_current_actor(state) or "旁白 / 自由行动"

def writing_mode_label(state):
    return "故事模式 | 导演评价中"

def command_help_text():
    return (
        "### 指令速查\n"
        "- **场景设定**：在「场景简描」中简要描述出场角色、地点和开场状况，点击「快速开始」即可直接进入故事模式；留空点击则无大纲开始，由你第一段输入指定开场。\n"
        "- **写作模式**：AI 自行推演一小段事件，你作为导演评价；可用下方「回复字数上限」滑块控制每次生成的字数。\n"
        "- `/k`：**通过当前事件**，AI 自动保存并继续推演下一事件。\n"
        "- `/s`：**场景大纲模式**，AI 自行生成当前场景的完整大纲（8-14条条目列表）。\n"
        "- `/u 你的修正`：上帝指令，用来修正剧情错误，不消耗回合。\n"
        "- `/c角色设定修改`：临时更新或创建角色卡；确认后写入角色卡并重新载入，不写入剧情记录。\n"
        "- `/w世界设定修改`：临时更新世界设定；确认后写入 World.txt 并重新载入，不写入剧情记录。\n"
        "- **评价/反馈**：输入你的想法，AI 会参考后重新生成当前事件。\n"
        "- `结束本幕`：AI 先生成场景大纲并保存，再进行结算（角色记忆/世界观）。\n"
    )

def sanitize_codename(raw_name):
    cn = "".join(c if c.isalnum() else "_" for c in raw_name.strip().lower())
    cn = re.sub(r"_+", "_", cn).strip("_")
    return cn or "unknown_character"

def unique_character_codename(sdir, codename):
    cdir = ensure_char_dir(sdir)
    base = sanitize_codename(codename)
    candidate = base
    index = 1
    while os.path.exists(os.path.join(cdir, f"{candidate}.txt")):
        candidate = f"{base}_{index}"
        index += 1
    return candidate

def extract_json_object(raw_response):
    if not raw_response: return {}
    raw_response = raw_response.strip()
    raw_response = re.sub(r'^```(?:json)?\s*\n?', '', raw_response)
    raw_response = re.sub(r'\n?```\s*$', '', raw_response).strip()
    m = re.search(r'\{.*\}', raw_response, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else raw_response)
    except json.JSONDecodeError:
        return {}

def limit_words(text, max_words=200):
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()

def limit_chars(text, max_chars=200):
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].strip()

CHARACTER_FIELDS = ["性别", "年龄", "职业", "外貌", "性格特征", "爱好", "过往经历", "人际关系", "能力"]

def extract_character_id(profile):
    match = re.search(r"姓名\s*[:：]\s*.*?[（(]\s*([A-Za-z0-9_]+)\s*[）)]", profile or "")
    return sanitize_codename(match.group(1)) if match else ""

def extract_character_name(profile):
    match = re.search(r"姓名\s*[:：]\s*([^（(\n]+)", profile or "")
    return match.group(1).strip() if match else ""

def local_codename_from_text(text, fallback="new_character"):
    """Create a local filename id without calling an API."""
    text = (text or "").strip()
    ascii_part = "".join(c.lower() if c.isalnum() else "_" for c in text if c.isascii())
    ascii_part = re.sub(r"_+", "_", ascii_part).strip("_")
    if ascii_part:
        return sanitize_codename(ascii_part)
    return sanitize_codename(fallback)

def normalize_character_profile(profile, codename, fallback_name="未命名角色"):
    """Force saved/generated character profiles into the project's required format."""
    codename = sanitize_codename(codename)
    values = {field: "" for field in CHARACTER_FIELDS}
    name = extract_character_name(profile) or fallback_name
    current_field = None
    field_pattern = re.compile(rf"^({'|'.join(CHARACTER_FIELDS)})\s*[:：]?\s*(.*)$")

    for raw_line in (profile or "").replace("\r\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("姓名"):
            parsed_name = extract_character_name(line)
            if parsed_name:
                name = parsed_name
            current_field = None
            continue
        match = field_pattern.match(line)
        if match:
            current_field = match.group(1)
            values[current_field] = match.group(2).strip()
        elif current_field:
            values[current_field] = (values[current_field] + " " + line).strip()
        else:
            values["过往经历"] = (values["过往经历"] + " " + line).strip()

    return (
        f"姓名：{name}（{codename}）\n"
        f"性别：{values['性别']}\n"
        f"年龄：{values['年龄']}\n"
        f"职业：{values['职业']}\n"
        f"外貌：{values['外貌']}\n\n"
        f"性格特征：{values['性格特征']}\n"
        f"爱好：{values['爱好']}\n"
        f"过往经历：{values['过往经历']}\n"
        f"人际关系：{values['人际关系']}\n"
        f"能力：{values['能力']}"
    ).strip()

def normalize_model(model):
    model = (model or DEFAULT_MODEL).strip()
    return model if model in MODEL_CHOICES else DEFAULT_MODEL

def normalize_thinking_type(thinking_type):
    thinking_type = (thinking_type or DEFAULT_THINKING_TYPE).strip().lower()
    return thinking_type if thinking_type in THINKING_TYPE_CHOICES else DEFAULT_THINKING_TYPE

def normalize_reasoning_effort(effort):
    effort = (effort or DEFAULT_REASONING_EFFORT).strip().lower()
    if effort in ("low", "medium", "high"):
        return "high"
    if effort in ("xhigh", "max"):
        return "max"
    return DEFAULT_REASONING_EFFORT

def sync_api_settings(state):
    global _global_thinking_type, _global_reasoning_effort
    if not isinstance(state, dict):
        return DEFAULT_MODEL, DEFAULT_THINKING_TYPE, DEFAULT_REASONING_EFFORT
    model = normalize_model(state.get("model", DEFAULT_MODEL))
    thinking_type = normalize_thinking_type(state.get("thinking_type", DEFAULT_THINKING_TYPE))
    reasoning_effort = normalize_reasoning_effort(state.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
    state["model"] = model
    state["thinking_type"] = thinking_type
    state["reasoning_effort"] = reasoning_effort
    _global_thinking_type = thinking_type
    _global_reasoning_effort = reasoning_effort
    return model, thinking_type, reasoning_effort

def get_client(state):
    """Get or create the OpenAI client using the API key from state."""
    global _global_client, _global_api_key
    sync_api_settings(state)
    key = state.get("api_key", "").strip() if state else ""
    if not key:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    
    if key != _global_api_key or _global_client is None:
        _global_api_key = key
        _global_client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return _global_client

def _message_content_text(message):
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content = "".join(parts)
    return str(content or "").strip()

def call_deepseek(client, model, msgs, max_tok=1000):
    if client is None:
        return None, "请先在「启动 / API」里输入 DeepSeek API Key。"
    model = normalize_model(model)
    thinking_type = normalize_thinking_type(_global_thinking_type)
    reasoning_effort = normalize_reasoning_effort(_global_reasoning_effort)
    request_kwargs = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tok,
        "extra_body": {"thinking": {"type": thinking_type}},
    }
    if thinking_type == "enabled":
        request_kwargs["reasoning_effort"] = reasoning_effort
    else:
        request_kwargs["temperature"] = 0.7
    try:
        for attempt in range(3):
            r = client.chat.completions.create(**request_kwargs)
            content = _message_content_text(r.choices[0].message) if r.choices else ""
            if content:
                return content, None
            if attempt < 2:
                continue
        return None, "AI 返回空内容，请重试；如果反复出现，请减少回复字数或换一个模型。"
    except AuthenticationError: return None, "认证失败，请检查 DeepSeek API Key。"
    except RateLimitError: return None, "请求过于频繁，请稍后再试。"
    except APIError as e: return None, f"API 错误：{e}"
    except Exception as e: return None, f"错误：{e}"

def make_state():
    return {"story_dir":DEFAULT_STORY_DIR,"api_key":"","model":DEFAULT_MODEL,
        "thinking_type":DEFAULT_THINKING_TYPE,"reasoning_effort":DEFAULT_REASONING_EFFORT,
        "world_bg":"","characters":{},"scene_num":1,"scene_name":"scene_01.txt",
        "scene_path":os.path.join(DEFAULT_STORY_DIR,"scene_01.txt"),
        "scene_premise":"","scene_outline":"","active_characters":[],"mode":"PLANNING",
        "turn_order":[],"turn_index":0,
        "planning_history":[],"planning_turns":0,"writing_history":[],
        "current_draft":"","pending_user_input":"","pending_turn_input":"","pending_actor":"",
        "pending_new_characters":[],"reply_char_count":DEFAULT_REPLY_CHARS,
        "rejected_draft_history":[],"revision_ui_start":None,"undo_stack":[],
        "pending_setting_update":None,"scene_summary":"","scene_summary_history_count":0,
        "fact_pack":{},"fact_pack_history_count":-1,"fact_pack_scene":"",
        "accepted_event_total":0}

def _clear_revision_state(state):
    state["rejected_draft_history"] = []
    state["revision_ui_start"] = None
    return state

def _clear_pending_draft(state):
    state["current_draft"] = ""
    state["pending_user_input"] = ""
    state["pending_turn_input"] = ""
    state["pending_actor"] = ""
    state["pending_setting_update"] = None
    return state

def _trim_writing_history(state):
    if len(state.get("writing_history", [])) > MAX_HISTORY_PAIRS * 2:
        state["writing_history"] = state["writing_history"][-MAX_HISTORY_PAIRS * 2:]
    return state

def _clear_undo_stack(state):
    state["undo_stack"] = []
    return state

def _recent_writing_history_window(state):
    """Return recent formal story messages, newest-first budgeted like chat UIs."""
    history = [dict(m) for m in (state.get("writing_history", []) or []) if str(m.get("content", "")).strip()]
    recent = history[-RECENT_CONTEXT_MESSAGES:]
    selected = []
    used_chars = 0
    for msg in reversed(recent):
        content_len = len(str(msg.get("content", "")))
        if selected and used_chars + content_len > STORY_HISTORY_CONTEXT_CHARS:
            break
        selected.append(msg)
        used_chars += content_len
    return list(reversed(selected))

def _story_history_message_for_api(msg):
    role = msg.get("role") if msg.get("role") in ("user", "assistant") else "user"
    label = "导演" if role == "user" else "剧情"
    content = str(msg.get("content", "")).strip()
    return {"role": role, "content": f"[{label}]\n{content}"}

def _api_writing_context(state):
    """What normal chat windows do: keep summaries plus a small recent message window."""
    return [_story_history_message_for_api(m) for m in _recent_writing_history_window(state)]

def _past_outline(state):
    return read_file(outline_path(current_story_dir(state)))

def _outline_context_block(state):
    outline = _past_outline(state)
    return f"\n\n[过去章节提纲]\n{outline}" if outline else ""

def _history_message_count(state):
    return len(state.get("writing_history", []) or [])

def _accepted_event_count(state):
    return sum(1 for m in state.get("writing_history", []) if m.get("role") == "assistant")

def _reset_scene_support_context(state):
    state["scene_summary"] = ""
    state["scene_summary_history_count"] = 0
    state["fact_pack"] = {}
    state["fact_pack_history_count"] = -1
    state["fact_pack_scene"] = ""
    state["accepted_event_total"] = 0
    state["_last_fact_check_issues"] = []
    return state

def _character_field(profile, field):
    match = re.search(rf"^{re.escape(field)}\s*[:：]\s*(.*)$", profile or "", re.MULTILINE)
    return match.group(1).strip() if match else ""

def _character_key_fact_lines(state):
    chars = state.get("characters", {}) or {}
    targets = state.get("active_characters", []) or sorted(chars.keys())
    lines = []
    sdir = current_story_dir(state)
    for cn in targets:
        profile = read_file(character_path(sdir, cn)) or chars.get(cn, "")
        if not profile:
            continue
        name = extract_character_name(profile) or cn
        parts = [f"{name}（{cn}）"]
        for field in ["性别", "年龄", "职业", "人际关系", "能力"]:
            value = _character_field(profile, field)
            if value:
                parts.append(f"{field}：{value}")
        lines.append("；".join(parts))
    return lines

def _local_fact_pack(state):
    immutable = _character_key_fact_lines(state)
    current = []
    if state.get("scene_premise"):
        current.append(f"本幕场景：{state.get('scene_premise')}")
    if state.get("scene_summary"):
        current.append(f"当前场景总结：{state.get('scene_summary')}")
    forbidden = [
        "不得改写角色姓名、真实年龄、身份关系、职业和已设定能力。",
        "不得让角色突然拥有角色卡中不存在的知识、能力或关系。",
        "不得把已接受剧情改写成未发生或相反事实。",
    ]
    return {
        "immutable_facts": immutable,
        "current_state": current,
        "forbidden_contradictions": forbidden,
    }

def _clean_fact_pack(data, fallback):
    if not isinstance(data, dict):
        data = {}
    cleaned = {}
    for key in ["immutable_facts", "current_state", "forbidden_contradictions"]:
        value = data.get(key, [])
        if isinstance(value, str):
            items = [line.strip("-• \t") for line in value.splitlines() if line.strip()]
        elif isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = []
        base = fallback.get(key, []) if isinstance(fallback, dict) else []
        merged = []
        for item in list(base) + items:
            if item and item not in merged:
                merged.append(item)
        cleaned[key] = merged[:24]
    return cleaned

def _fact_pack_text(state):
    pack = state.get("fact_pack") or {}
    if not isinstance(pack, dict) or not any(pack.get(k) for k in ["immutable_facts", "current_state", "forbidden_contradictions"]):
        return ""
    labels = [
        ("immutable_facts", "不可改事实"),
        ("current_state", "当前状态"),
        ("forbidden_contradictions", "禁止矛盾"),
    ]
    parts = []
    for key, label in labels:
        items = pack.get(key, [])
        if items:
            parts.append(f"{label}：\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(parts)

def _history_text(state):
    lines = []
    for idx, msg in enumerate(state.get("writing_history", []) or [], 1):
        label = "导演" if msg.get("role") == "user" else "剧情"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{idx}. {label}: {content}")
    return "\n".join(lines)

def _fallback_full_scene_summary(state, max_chars=1200):
    events = []
    index = 1
    for msg in state.get("writing_history", []) or []:
        if msg.get("role") != "assistant":
            continue
        content = " ".join(str(msg.get("content", "")).split())
        if content:
            events.append(f"事件{index}：{limit_chars(content, 140)}")
            index += 1
    return limit_chars("；".join(events), max_chars)

def _compact_writing_history_after_summary(state):
    history = state.get("writing_history", []) or []
    if len(history) <= RECENT_CONTEXT_MESSAGES:
        return 0
    remove_count = (len(history) * 2) // 3
    if remove_count <= 0:
        return 0
    state["writing_history"] = [dict(m) for m in history[remove_count:]]
    return remove_count

def _update_scene_summary_if_needed(state, client, force=False):
    history = state.get("writing_history", []) or []
    count = int(state.get("accepted_event_total") or _accepted_event_count(state))
    if not history:
        state["scene_summary"] = ""
        state["scene_summary_history_count"] = 0
        return False
    if not force and len(history) <= RECENT_CONTEXT_MESSAGES:
        return False
    if not force and state.get("scene_summary") and state.get("scene_summary_history_count") == count:
        return False
    fallback = _fallback_full_scene_summary(state)
    if client is None:
        state["scene_summary"] = fallback
        state["scene_summary_history_count"] = count
        _compact_writing_history_after_summary(state)
        return bool(fallback)
    system_prompt = (
        "你是互动小说当前场景的连续性整理器。请根据正式通过的本幕记录，写一个当前场景总结。\n"
        "要求：覆盖从开场到现在的所有已发生事件；按时间顺序简述；保留角色状态、地点、目标、矛盾、已造成的后果；"
        "不要评价文风，不要续写剧情。控制在 1200 个中文字符以内。"
    )
    user_prompt = (
        f"[本幕场景]\n{state.get('scene_premise','')}\n\n"
        f"[已有场景总结]\n{state.get('scene_summary','') or '无'}\n\n"
        f"[正式通过记录]\n{_history_text(state)}"
    )
    raw, err = call_deepseek(client, state["model"], [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt},
    ], max_tok=MAX_TOKENS_SCENE_MEMORY)
    summary = limit_chars(raw or fallback, 1200)
    if summary:
        state["scene_summary"] = summary
        state["scene_summary_history_count"] = count
        _compact_writing_history_after_summary(state)
        return True
    return False

def _refresh_fact_pack(state, client, force=False):
    count = int(state.get("accepted_event_total") or _accepted_event_count(state))
    scene_name = state.get("scene_name", "")
    if (not force and state.get("fact_pack") and
            state.get("fact_pack_history_count") == count and
            state.get("fact_pack_scene") == scene_name):
        return state.get("fact_pack")
    fallback = _local_fact_pack(state)
    if client is None:
        state["fact_pack"] = fallback
        state["fact_pack_history_count"] = count
        state["fact_pack_scene"] = scene_name
        return fallback
    char_context = _build_character_context(state, active_only=True) or _build_character_context(state, active_only=False)
    system_prompt = (
        "你是互动小说事实管理员。请从世界观、过去章节提纲、角色卡、本幕场景和已接受剧情中提取事实包。\n"
        "只输出合法 JSON，不要 markdown。格式："
        '{"immutable_facts":["不可改事实"],"current_state":["当前状态"],"forbidden_contradictions":["禁止矛盾"]}\n'
        "重点提取：真实年龄、身份、亲属/阵营关系、职业、能力限制、地点、时间线、已确认事件和当前状态。"
    )
    user_prompt = (
        f"[世界观]\n{state.get('world_bg','')}\n\n"
        f"[过去章节提纲]\n{_past_outline(state)}\n\n"
        f"[本幕场景]\n{state.get('scene_premise','')}\n\n"
        f"[当前场景总结]\n{state.get('scene_summary','')}\n\n"
        f"[角色卡]\n{char_context}\n\n"
        f"[正式通过记录]\n{_history_text(state)}"
    )
    raw, err = call_deepseek(client, state["model"], [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt},
    ], max_tok=MAX_TOKENS_FACT_PACK)
    data = extract_json_object(raw or "") if not err else {}
    pack = _clean_fact_pack(data, fallback)
    state["fact_pack"] = pack
    state["fact_pack_history_count"] = count
    state["fact_pack_scene"] = scene_name
    return pack

def _refresh_scene_support_context(state, client, force=False):
    _update_scene_summary_if_needed(state, client, force=force)
    _refresh_fact_pack(state, client, force=force)
    return state

def _fact_check_draft(state, client, draft):
    fact_text = _fact_pack_text(state)
    if client is None or not draft or not fact_text:
        return True, [], ""
    system_prompt = (
        "你是互动小说事实校验器。只检查草稿是否违反事实包、角色卡、当前场景总结和已接受剧情。"
        "不要评价文风，不要提出新剧情。只输出合法 JSON："
        '{"pass":true,"issues":[],"rewrite_instruction":""}'
    )
    user_prompt = (
        f"[事实包]\n{fact_text}\n\n"
        f"[当前场景总结]\n{state.get('scene_summary','') or '无'}\n\n"
        f"[最近正式剧情]\n{_recent_story_context_text(state)}\n\n"
        f"[待检查草稿]\n{draft}"
    )
    raw, err = call_deepseek(client, state["model"], [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt},
    ], max_tok=MAX_TOKENS_FACT_CHECK)
    if err or not raw:
        return True, [], ""
    data = extract_json_object(raw)
    if not data:
        return True, [], ""
    passed = data.get("pass", True)
    if isinstance(passed, str):
        passed = passed.strip().lower() in ("true", "yes", "1", "pass", "passed")
    issues = data.get("issues", [])
    if isinstance(issues, str):
        issues = [line.strip("-• \t") for line in issues.splitlines() if line.strip()]
    elif isinstance(issues, list):
        issues = [str(item).strip() for item in issues if str(item).strip()]
    else:
        issues = []
    instruction = str(data.get("rewrite_instruction", "") or "").strip()
    return bool(passed), issues, instruction

def _validate_and_rewrite_draft(state, client, draft, original_msgs, max_tok, attempts=MAX_FACT_REWRITE_ATTEMPTS):
    if not draft:
        return draft
    state["_last_fact_check_issues"] = []
    for _ in range(attempts):
        passed, issues, instruction = _fact_check_draft(state, client, draft)
        if passed:
            return draft
        state["_last_fact_check_issues"] = issues
        issue_text = "\n".join(f"- {item}" for item in issues) if issues else "- 草稿与事实包存在冲突。"
        rewrite_instruction = instruction or "请重写当前事件，只修正事实矛盾，保持原本剧情意图和篇幅。"
        rewrite_msgs = [dict(m) for m in original_msgs]
        rewrite_msgs.append({"role":"assistant","content":draft})
        rewrite_msgs.append({
            "role":"user",
            "content": f"事实校验发现以下问题：\n{issue_text}\n\n{rewrite_instruction}\n只输出修正后的故事正文。",
        })
        rewritten, err = call_deepseek(client, state["model"], rewrite_msgs, max_tok=max_tok)
        if err or not rewritten:
            return draft
        draft = rewritten
    return draft

def _fallback_scene_summary_from_history(state, max_chars=300):
    events = [m.get("content", "").strip() for m in state.get("writing_history", []) if m.get("role") == "assistant" and m.get("content")]
    if not events:
        return ""
    joined = " / ".join(limit_chars(e.replace("\n", " "), 120) for e in events[-3:])
    return limit_chars(joined, max_chars)

def _update_outline_file(state, scene_name, summary):
    summary = limit_chars(" ".join((summary or "").split()), 500)
    if not summary:
        return False
    sdir = current_story_dir(state)
    fp = outline_path(sdir)
    key = os.path.splitext(scene_name or state.get("scene_name", "scene_01.txt"))[0]
    line = f"{key}: {summary}"
    lines = [ln.rstrip() for ln in read_file_raw(fp).splitlines() if ln.strip()]
    replaced = False
    for idx, existing in enumerate(lines):
        if existing.startswith(f"{key}:"):
            lines[idx] = line
            replaced = True
            break
    if not replaced:
        lines.append(line)
    write_file(fp, "\n".join(lines))
    return True

def _push_undo_snapshot(state, wcb=None):
    scene_path = state.get("scene_path", "")
    if state.get("revision_ui_start") is not None:
        ui_start = state.get("revision_ui_start")
    elif wcb and wcb[-1].get("role") == "assistant":
        ui_start = max(len(wcb) - 1, 0)
    else:
        ui_start = len(wcb or [])
    snapshot = {
        "scene_path": scene_path,
        "scene_before": read_file_raw(scene_path) if scene_path else "",
        "writing_history_before": [dict(m) for m in state.get("writing_history", [])],
        "scene_summary_before": state.get("scene_summary", ""),
        "scene_summary_history_count_before": state.get("scene_summary_history_count", 0),
        "fact_pack_before": json.loads(json.dumps(state.get("fact_pack", {}), ensure_ascii=False)),
        "fact_pack_history_count_before": state.get("fact_pack_history_count", -1),
        "fact_pack_scene_before": state.get("fact_pack_scene", ""),
        "accepted_event_total_before": state.get("accepted_event_total", _accepted_event_count(state)),
        "ui_start": ui_start,
    }
    stack = state.setdefault("undo_stack", [])
    stack.append(snapshot)
    if len(stack) > UNDO_STACK_LIMIT:
        del stack[:-UNDO_STACK_LIMIT]
    return snapshot

def _remember_rejected_current_draft(state, wcb, remove_from_ui=True):
    draft = (state.get("current_draft") or "").strip()
    if not draft:
        return False
    first_rejection = state.get("revision_ui_start") is None
    revision_history = state.setdefault("rejected_draft_history", [])
    if first_rejection:
        if wcb and wcb[-1].get("role") == "assistant":
            ui_start = max(len(wcb) - 1, 0)
            if len(wcb) >= 2 and wcb[-2].get("role") == "user":
                ui_start = max(len(wcb) - 2, 0)
            state["revision_ui_start"] = ui_start
        else:
            state["revision_ui_start"] = len(wcb or [])
        original_prompt = (state.get("pending_turn_input") or state.get("pending_user_input") or "").strip()
        original_label = (state.get("pending_user_input") or "").strip()
        if original_prompt:
            if original_label and original_label != original_prompt:
                original_prompt = f"[原始导演输入]\n{original_label}\n\n[原始生成请求]\n{original_prompt}"
            else:
                original_prompt = f"[原始导演输入]\n{original_prompt}"
            revision_history.append({
                "role": "user",
                "content": original_prompt
            })
    revision_history.append({
        "role": "assistant",
        "content": draft
    })
    if remove_from_ui and wcb and wcb[-1].get("role") == "assistant":
        wcb.pop()
    _clear_pending_draft(state)
    return True

def _revision_feedback_message(command):
    command = (command or "").strip()
    if not command or command.lower() == "/k":
        feedback = "请重写当前事件。"
    else:
        feedback = command
    return {
        "role": "user",
        "content": (
            "[修改意见]\n"
            f"{feedback}\n\n"
            "请基于上一版和修改意见重写当前事件。"
        )
    }

def _commit_current_draft(state, wcb=None):
    if not state.get("current_draft"):
        return None
    _push_undo_snapshot(state, wcb)
    ui = state.get("pending_user_input","")
    turn_input = state.get("pending_turn_input","") or ui
    actor = state.get("pending_actor","")
    draft = state.get("current_draft","")
    is_god = actor == "__god__"
    is_skip = actor == "__skip__"
    story_label = "CORRECTION" if is_god else ("SCENE SYNOPSIS" if is_skip else "EVENT")
    had_revision = bool(state.get("rejected_draft_history"))
    revision_ui_start = state.get("revision_ui_start")
    if had_revision:
        append_file(state.get("scene_path",""), f"**{story_label}:**\n{draft}")
        state["writing_history"].append({"role":"assistant","content":draft})
    else:
        actor_label = "GOD COMMAND" if is_god else "DIRECTOR"
        append_file(state.get("scene_path",""), f"**{actor_label}:**\n{ui}\n\n**{story_label}:**\n{draft}")
        state["writing_history"].append({"role":"user","content":turn_input})
        state["writing_history"].append({"role":"assistant","content":draft})
    state["accepted_event_total"] = int(state.get("accepted_event_total") or max(_accepted_event_count(state) - 1, 0)) + 1
    _trim_writing_history(state)
    _clear_pending_draft(state)
    _clear_revision_state(state)
    return {"draft": draft, "is_skip": is_skip, "had_revision": had_revision, "revision_ui_start": revision_ui_start}

def _replace_revision_ui_with_accepted(wcb, commit_info, title="事件已保存"):
    if not commit_info or not commit_info.get("had_revision"):
        return wcb
    start = commit_info.get("revision_ui_start")
    if isinstance(start, int) and 0 <= start <= len(wcb):
        del wcb[start:]
    wcb.append({"role":"assistant","content":f"*{title}（已清理废弃版本）：*\n\n{commit_info.get('draft','')}"})
    return wcb

def _scene_entry_start(raw, heading_start):
    idx = heading_start
    while idx > 0 and raw[idx - 1].isspace():
        idx -= 1
    return idx

def _scene_record_blocks(raw):
    pattern = re.compile(
        r"(?m)^\s*\*\*(DIRECTOR|EVENT|GOD COMMAND|CORRECTION|SCENE SYNOPSIS):\*\*\s*$"
        r"|^\s*\[(Scene Summary)\]\s*$"
    )
    matches = list(pattern.finditer(raw or ""))
    if not matches:
        content = (raw or "").strip()
        return [{"label":"EVENT","content":content,"start":0}] if content else []

    blocks = []
    prefix = (raw[:matches[0].start()] or "").strip()
    if prefix:
        blocks.append({"label":"EVENT","content":prefix,"start":0})
    for idx, match in enumerate(matches):
        label = match.group(1) or match.group(2)
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        content = raw[match.end():end].strip()
        blocks.append({"label":label,"content":content,"start":match.start()})
    return blocks

def _parse_scene_record(scene_path, scene_name):
    raw = read_file_raw(scene_path)
    blocks = _scene_record_blocks(raw)
    ui_messages = [{"role":"assistant","content":f"**{scene_name}**\n已载入场景记录，可直接继续；输入 /k 或评价即可推进。"}]
    history = []
    undo_snapshots = []
    first_story = ""
    summaries = []
    pending_user = None

    def remember_snapshot(scene_before, history_before, ui_start):
        undo_snapshots.append({
            "scene_path": scene_path,
            "scene_before": scene_before,
            "writing_history_before": [dict(m) for m in history_before],
            "ui_start": ui_start,
        })

    for block in blocks:
        label = block.get("label", "")
        content = (block.get("content") or "").strip()
        if not content:
            continue
        entry_start = _scene_entry_start(raw, block.get("start", 0))

        if label == "Scene Summary":
            summaries.append(content)
            continue

        if label in ("DIRECTOR", "GOD COMMAND"):
            user_content = content
            if label == "GOD COMMAND" and not content.lower().startswith("/u"):
                user_content = f"[上帝指令]\n{content}"
            pending_user = {
                "scene_before": raw[:entry_start].rstrip(),
                "history_before": [dict(m) for m in history],
                "ui_start": len(ui_messages),
            }
            history.append({"role":"user","content":user_content})
            ui_messages.append({"role":"user","content":content})
            continue

        if label in ("EVENT", "CORRECTION", "SCENE SYNOPSIS"):
            if not first_story:
                first_story = content
            if pending_user:
                remember_snapshot(
                    pending_user["scene_before"],
                    pending_user["history_before"],
                    pending_user["ui_start"]
                )
            else:
                remember_snapshot(raw[:entry_start].rstrip(), history, len(ui_messages))
            history.append({"role":"assistant","content":content})
            ui_messages.append({"role":"assistant","content":content})
            pending_user = None

    if len(undo_snapshots) > UNDO_STACK_LIMIT:
        undo_snapshots = undo_snapshots[-UNDO_STACK_LIMIT:]
    return {
        "raw": raw,
        "ui_messages": ui_messages,
        "writing_history": history,
        "undo_stack": undo_snapshots,
        "first_story": first_story,
        "summaries": summaries,
    }

def _outline_summary_for_scene(state, scene_name):
    key = os.path.splitext(scene_name or "")[0]
    for line in read_file_raw(outline_path(current_story_dir(state))).splitlines():
        m = re.match(rf"^{re.escape(key)}\s*[:：]\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return ""

def _scene_premise_from_loaded_record(state, scene_name, parsed):
    outline_summary = _outline_summary_for_scene(state, scene_name)
    if outline_summary:
        return f"继续已有场景：{outline_summary}"
    first_story = (parsed.get("first_story") or "").replace("\n", " ")
    if first_story:
        return f"继续已有场景：{limit_chars(first_story, 180)}"
    return "继续已有场景"

def _active_characters_for_scene(state, raw):
    chars = state.get("characters", {}) or {}
    matched = []
    for cn in sorted(chars.keys()):
        profile = chars.get(cn, "")
        name = extract_character_name(profile)
        probes = [cn, name]
        if name and len(name) >= 3:
            probes.append(name[-2:])
        if any(probe and probe in raw for probe in probes):
            matched.append(cn)
    return matched or sorted(chars.keys())

# ---- Setup Handlers ----
def load_characters_from_char_dir(state):
    """Read every .txt file from the active story directory's char folder."""
    sdir = current_story_dir(state)
    cdir = ensure_char_dir(sdir)
    characters = {}
    for fn in sorted(os.listdir(cdir)):
        if fn.endswith(".txt") and not fn.startswith("."):
            cn = os.path.splitext(fn)[0]
            characters[cn] = read_file(os.path.join(cdir, fn))
    state["characters"] = characters
    return list(characters.keys())

def handle_init(state):
    sdir = current_story_dir(state)
    os.makedirs(sdir, exist_ok=True)
    wbg = read_file(os.path.join(sdir,"World.txt"))
    choices = load_characters_from_char_dir(state)
    chars = state.get("characters", {})
    sp, sn, snum = get_current_scene_file(sdir)
    for k in ["world_bg","characters","scene_num","scene_name","scene_path",
              "scene_premise","active_characters"]:
        state[k] = (wbg if k=="world_bg" else chars if k=="characters" else snum if k=="scene_num"
        else sn if k=="scene_name" else sp if k=="scene_path" else "" if k=="scene_premise" else [])
    state["turn_order"] = []
    state["turn_index"] = 0
    state.update({"mode":"PLANNING","planning_history":[],"planning_turns":0,
        "writing_history":[],"current_draft":"","pending_user_input":"",
        "pending_turn_input":"","pending_actor":"","pending_new_characters":[],"scene_outline":""})
    _clear_revision_state(state)
    _clear_undo_stack(state)
    _reset_scene_support_context(state)
    ws = f"世界观: {'已载入' if wbg else '未找到'} | 角色数: {len(chars)} | 下一幕: {sn}"
    wbg_text = wbg if wbg else ""
    wmsg_val = "已载入现有世界观" if wbg else "尚未创建世界观，可输入提示词后点击生成"
    return state, ws, f"初始化完成：{ws}", gr.update(choices=list(chars.keys()), value=None), wbg_text, wmsg_val

def handle_refresh_chars(state):
    choices = load_characters_from_char_dir(state)
    return state, gr.update(choices=choices, value=None)

def handle_refresh_scene_choices(state):
    scenes = list_scene_files(current_story_dir(state))
    value = scenes[-1] if scenes else None
    msg = f"已找到 {len(scenes)} 个已有场景。" if scenes else "当前故事目录没有可继续的场景。"
    return gr.update(choices=scenes, value=value), msg

def handle_scene_dir_choices(nd):
    scenes = list_scene_files(normalize_story_dir(nd))
    return gr.update(choices=scenes, value=scenes[-1] if scenes else None)

def handle_continue_scene(state, selected_scene):
    sdir = current_story_dir(state)
    scenes = list_scene_files(sdir)
    if not selected_scene or selected_scene not in scenes:
        wcb = [{"role":"assistant","content":"请选择一个已有场景文件。"}]
        return state, wcb, "导演会议", f"当前幕: {state.get('scene_name','scene_01.txt')} | 场景: 待定", gr.update(visible=True), gr.update(visible=False), "请选择一个已有场景文件。"

    # Re-read world and characters from disk every time a scene starts.
    wbg = read_file(os.path.join(sdir, "World.txt"))
    state["world_bg"] = wbg
    load_characters_from_char_dir(state)

    scene_path = os.path.join(sdir, selected_scene)
    parsed = _parse_scene_record(scene_path, selected_scene)
    premise = _scene_premise_from_loaded_record(state, selected_scene, parsed)
    active = _active_characters_for_scene(state, parsed.get("raw", ""))

    state.update({
        "scene_num": scene_number_from_name(selected_scene),
        "scene_name": selected_scene,
        "scene_path": scene_path,
        "scene_premise": premise,
        "scene_outline": premise,
        "active_characters": active,
        "turn_order": build_turn_order(active),
        "turn_index": 0,
        "mode": "WRITING",
        "planning_history": [],
        "planning_turns": 0,
        "writing_history": parsed.get("writing_history", []),
        "current_draft": "",
        "pending_user_input": "",
        "pending_turn_input": "",
        "pending_actor": "",
        "pending_new_characters": [],
        "undo_stack": parsed.get("undo_stack", []),
    })
    _clear_revision_state(state)
    _reset_scene_support_context(state)
    state["accepted_event_total"] = _accepted_event_count(state)
    if parsed.get("summaries"):
        state["scene_summary"] = parsed.get("summaries", [""])[-1]
        state["scene_summary_history_count"] = state["accepted_event_total"]

    record_count = len(parsed.get("writing_history", []))
    chars_str = ", ".join(active) if active else "未指定"
    wcb = parsed.get("ui_messages", [])
    wcb.append({"role":"assistant","content":f"*已跳过导演会议，继续 {selected_scene}。已恢复 {record_count} 条正式记录。登场角色: {chars_str}。*"})
    if parsed.get("summaries"):
        wcb.append({"role":"assistant","content":"*提示：该场景文件已包含结算摘要，如需继续请确认这是你想续写的版本。*"})
    st = f"当前幕: {selected_scene} | 场景: {premise[:60]}..."
    return state, wcb, writing_mode_label(state), st, gr.update(visible=False), gr.update(visible=True), f"已继续 {selected_scene}。"

def handle_change_dir(state, nd): state["story_dir"] = normalize_story_dir(nd); return state
def handle_change_key(state, nk):
    global _global_client; state["api_key"] = nk; _global_client = None; return state
def handle_change_model(state, nm):
    state["model"] = normalize_model(nm)
    sync_api_settings(state)
    return state
def handle_change_thinking_type(state, thinking_type):
    state["thinking_type"] = normalize_thinking_type(thinking_type)
    sync_api_settings(state)
    return state
def handle_change_reasoning_effort(state, effort):
    state["reasoning_effort"] = normalize_reasoning_effort(effort)
    sync_api_settings(state)
    return state

# ---- World Handlers ----
def handle_world_generate_or_revise(state, concept, wdisp):
    """Generate or revise world. If world_bg is set, treat concept as feedback for modification."""
    sdir = current_story_dir(state)
    if not concept or not concept.strip():
        return state, wdisp, "请输入世界设定或修改意见。"
    c = get_client(state)
    if c is None:
        return state, wdisp, "请先在「启动 / API」里输入 DeepSeek API Key。"
    wbg = state.get("world_bg", "")
    if wbg:
        # ---- MODIFY MODE ----
        msgs = [
            {"role":"system","content":"你是专业小说世界观设计师。根据反馈修改已有世界观，必须控制在 200 个中文字符以内。请用中文直接输出，不要寒暄。"},
            {"role":"assistant","content":wbg},
            {"role":"user","content":f"请根据以下意见修改世界观，结果控制在 200 中文字符以内：\n\n{concept.strip()}"}
        ]
        r, e = call_deepseek(c, state["model"], msgs, max_tok=400)
        if e: return state, wdisp, f"错误：{e}"
        r = limit_chars(r, 200)
        state["_wm"] = msgs + [{"role":"assistant","content":r}]
        return state, r, "已根据意见修改。请审阅后点击「接受/保存世界观」。"
    else:
        # ---- GENERATE MODE ----
        msgs = [
            {"role":"system","content":"你是专业小说世界观设计师。请用中文直接输出结构清晰的世界背景，不要寒暄。必须控制在 200 个中文字符以内。"},
            {"role":"user","content":concept.strip()}
        ]
        r, e = call_deepseek(c, state["model"], msgs, max_tok=400)
        if e: return state, wdisp, f"错误：{e}"
        r = limit_chars(r, 200)
        state["_wm"] = msgs + [{"role":"assistant","content":r}]
        return state, r, "已生成。请审阅后点击「接受/保存世界观」。"

def handle_save_world(state, wdisp):
    if not wdisp or not wdisp.strip(): return state, wdisp, "没有可保存的世界观。"
    sdir = current_story_dir(state)
    wdisp = limit_chars(wdisp, 200)
    write_file(os.path.join(sdir,"World.txt"), wdisp)
    state["world_bg"] = wdisp
    return state, wdisp, f"已保存到 {sdir}/World.txt（已确保 200 字以内）"

# ---- Character Handlers ----
def handle_generate_character(state, cdesc, cdisplay, char_name):
    if cdesc and cdesc.strip().lower() == "/k":
        choices = load_characters_from_char_dir(state)
        msg = f"已跳过角色生成，并载入 char 文件夹中的 {len(choices)} 个角色。"
        if "_editing_character_id" in state: del state["_editing_character_id"]
        if "_pending_character_id" in state: del state["_pending_character_id"]
        return state, cdisplay, char_name, msg, gr.update(choices=choices, value=None)
    if not cdesc or not cdesc.strip():
        return state, cdisplay, char_name, "请输入角色提示词；或输入 /k 直接使用当前 char 文件夹中的角色。", gr.update()
    c = get_client(state)
    if c is None:
        return state, cdisplay, char_name, "请先在「启动 / API」里输入 DeepSeek API Key。", gr.update()
    sdir = current_story_dir(state)
    wbg = state.get("world_bg","")
    editing_id = state.get("_editing_character_id", "")
    if editing_id:
        # ---- MODIFY MODE: a character is loaded, modify it per user's feedback ----
        existing_chars = _build_character_context(state, active_only=False)
        cn = editing_id
        sys_msg = (
            "你正在修改一个已有角色档案。必须严格保持字段格式：姓名、性别、年龄、职业、外貌、性格特征、爱好、过往经历、人际关系、能力。\n"
            f"姓名行里的 ID 必须保持为 {cn}。角色档案控制在 200 words 以内。"
        )
        if wbg:
            sys_msg += f"\n\n世界观：\n{wbg}"
        if existing_chars:
            sys_msg += f"\n\n[全部已有角色 —— 必须避免与之矛盾]\n{existing_chars}"
        msgs = [
            {"role":"system","content":sys_msg},
            {"role":"assistant","content":cdisplay.strip()},
            {"role":"user","content":f"请根据以下意见修改角色档案，保持字段格式，ID 为 {cn}：\n\n{cdesc.strip()}"}
        ]
        r, e = call_deepseek(c, state["model"], msgs, max_tok=700)
        if e: return state, cdisplay, char_name, f"错误：{e}", gr.update()
        state["_cm"] = msgs + [{"role":"assistant","content":r}]
        return state, r, char_name, f"已根据意见修改角色：{cn}", gr.update()
    else:
        # ---- GENERATE MODE: create new character(s) ----
        existing_chars = _build_character_context(state, active_only=False)
        is_batch = bool(re.search(r'(多个角色|以下几个|以下角色|生成.*个|批量|一群人|几个角色|小队|两个|三个|四个|五个|六个|一群|一批)', cdesc.strip()))
        sc = (
            "你是专业小说角色设计师。根据用户的提示词，生成一个或多个角色档案。\n"
            "用中文直接输出，不要寒暄。多角色之间用 === 分隔。\n"
            "每个角色档案必须严格使用下面字段和顺序：\n"
            "姓名：角色中文名（charN）\n性别：\n年龄：\n职业：\n外貌：\n\n"
            "性格特征：\n爱好：\n过往经历：\n人际关系：\n能力：\n\n"
            "姓名行括号里的 ID 统一使用 char1, char2, char3... 依次编号。\n"
            "性格特征和爱好写几个关键词；能力字段必须填写：普通人写「无」，有战斗/魔法/超能力的角色写具体能力描述。\n"
            "每个角色档案控制在 200 words 以内。角色之间不得互相矛盾。"
        )
        if wbg:
            sc += f"\n\n世界观：\n{wbg}"
        if existing_chars:
            sc += f"\n\n[已有角色 —— 必须避免与之矛盾]\n{existing_chars}"
        max_tok = 2500 if is_batch else 700
        r, e = call_deepseek(c, state["model"], [{"role":"system","content":sc},{"role":"user","content":cdesc.strip()}], max_tok=max_tok)
        if e: return state, cdisplay, char_name, f"错误：{e}", gr.update()
        state["_cm"] = [{"role":"system","content":sc},{"role":"user","content":cdesc.strip()},{"role":"assistant","content":r}]
        new_name = next_char_name(state)
        state["_editing_character_id"] = ""
        state["_pending_character_id"] = ""
        msg = "已生成角色。请审阅、设置文件名并保存。"
        if "===" in r:
            msg = "已生成多个角色（=== 分隔）。可分别编辑后依次保存，或点击「批量保存」一次性保存。"
        return state, r, new_name, msg, gr.update()

def next_char_name(state):
    """Find the next available charN name in the story directory."""
    sdir = current_story_dir(state)
    cdir = ensure_char_dir(sdir)
    existing = set()
    for fn in os.listdir(cdir):
        if fn.endswith(".txt") and not fn.startswith("."):
            existing.add(os.path.splitext(fn)[0])
    n = 1
    while f"char{n}" in existing:
        n += 1
    return f"char{n}"

def handle_load_character(state, sel, char_name):
    if not sel:
        return state, "", char_name, "请选择角色，或直接生成新角色。"
    cn = sel if isinstance(sel, str) else sel[0] if isinstance(sel, list) else str(sel)
    sdir = current_story_dir(state)
    fp = character_path(sdir, cn)
    profile = read_file(fp)
    if not profile:
        return state, "", char_name, f"找不到角色文件：{fp}"
    state.setdefault("characters",{})[cn] = profile
    state["_editing_character_id"] = cn
    state["_pending_character_id"] = ""
    wbg = state.get("world_bg","")
    existing_chars = _build_character_context(state, active_only=False)
    sys_msg = (
        "你正在修改一个已有角色档案。必须严格保持字段格式：姓名、性别、年龄、职业、外貌、性格特征、爱好、过往经历、人际关系、能力。"
        f"姓名行里的 ID 必须保持为 {cn}。角色档案必须控制在 200 words 以内。"
    )
    if wbg:
        sys_msg += f"\n\n世界观：\n{wbg}"
    if existing_chars:
        sys_msg += f"\n\n[全部已有角色 —— 必须避免与之矛盾]\n{existing_chars}"
    state["_cm"] = [
        {"role":"system","content": sys_msg},
        {"role":"assistant","content":profile}
    ]
    return state, profile, cn, f"已载入角色：{cn}"

def handle_save_character(state, char_name, cdisplay):
    if not cdisplay or not cdisplay.strip():
        return state, char_name, cdisplay, "没有可保存的角色。", gr.update()
    sdir = current_story_dir(state)
    ensure_char_dir(sdir)
    if not char_name or not char_name.strip():
        char_name = next_char_name(state)
    cn = sanitize_codename(char_name.strip())
    if not cn:
        cn = next_char_name(state)
    fp = character_path(sdir, cn)
    write_file(fp, cdisplay.strip())
    state["characters"][cn] = cdisplay.strip()
    state["_editing_character_id"] = cn
    state["_pending_character_id"] = ""
    choices = load_characters_from_char_dir(state)
    return state, cn, cdisplay, f"已保存角色：{cn}", gr.update(choices=choices, value=cn)

def handle_delete_character(state, sel):
    if not sel: return state, gr.update(), gr.update(), "请先选择角色。"
    cn = sel if isinstance(sel, str) else sel[0] if isinstance(sel, list) else str(sel)
    sdir = current_story_dir(state)
    fp = character_path(sdir, cn)
    if os.path.exists(fp): os.remove(fp)
    if cn in state["characters"]: del state["characters"][cn]
    if state.get("_editing_character_id") == cn:
        state["_editing_character_id"] = ""
    if state.get("_pending_character_id") == cn:
        state["_pending_character_id"] = ""
    # Return updated choices
    choices = list(state["characters"].keys())
    return state, gr.update(choices=choices, value=None), "", f"已删除角色文件：{cn}"

def handle_rename_character(state, new_name):
    """Rename a character file on disk. Uses the currently loaded character as the source."""
    old_cn = state.get("_editing_character_id", "")
    if not old_cn:
        return state, old_cn, "请先载入一个角色再重命名。", gr.update()
    if not new_name or not new_name.strip():
        return state, old_cn, "请输入新文件名。", gr.update()
    new_cn = sanitize_codename(new_name.strip())
    if not new_cn:
        return state, old_cn, "文件名无效。", gr.update()
    if old_cn == new_cn:
        return state, old_cn, f"文件名未变化：{old_cn}", gr.update()
    sdir = current_story_dir(state)
    old_fp = character_path(sdir, old_cn)
    new_fp = character_path(sdir, new_cn)
    if not os.path.exists(old_fp):
        return state, old_cn, f"找不到角色文件：{old_cn}", gr.update()
    if os.path.exists(new_fp):
        return state, old_cn, f"目标文件名 {new_cn} 已存在。", gr.update()
    os.rename(old_fp, new_fp)
    if old_cn in state.get("characters", {}):
        state["characters"][new_cn] = state["characters"].pop(old_cn)
    state["_editing_character_id"] = new_cn
    choices = load_characters_from_char_dir(state)
    return state, new_cn, f"已重命名：{old_cn} → {new_cn}", gr.update(choices=choices, value=new_cn)

def handle_batch_save_characters(state, cdisplay):
    """Split === separated character profiles and save each as charN."""
    if not cdisplay or not cdisplay.strip():
        return state, cdisplay, "没有可保存的角色。", gr.update()
    parts = [p.strip() for p in re.split(r'\n?===\n?', cdisplay.strip()) if p.strip()]
    if len(parts) <= 1:
        return state, cdisplay, "未检测到多个角色（用 === 分隔）。请使用「保存角色」保存单个角色。", gr.update()
    sdir = current_story_dir(state)
    ensure_char_dir(sdir)
    saved = []
    for part in parts:
        cn = next_char_name(state)
        fp = character_path(sdir, cn)
        write_file(fp, part)
        state["characters"][cn] = part
        saved.append(cn)
    state["_editing_character_id"] = ""
    state["_pending_character_id"] = ""
    choices = load_characters_from_char_dir(state)
    last = saved[-1] if saved else None
    return state, cdisplay, f"已批量保存 {len(saved)} 个角色：{', '.join(saved)}", gr.update(choices=choices, value=last)

# ---- Planning Handlers ----
def handle_planning_send(state, umsg, pcb):
    if not umsg or not umsg.strip():
        return state, pcb, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    wbg = state.get("world_bg","")
    sn = state.get("scene_name","scene_01.txt")
    command = umsg.strip()
    is_k = command.lower() == "/k"
    is_god = command.lower().startswith("/u")

    # /k in planning: AI auto-proposes scene and transitions to writing
    if is_k:
        kc = state.get("characters",{})
        existing_chars = list(kc.keys())
        # Build character profiles context so AI knows who the characters are
        char_profiles = _build_character_context(state, active_only=False)
        outline_block = _outline_context_block(state)
        sp = ('你是一个互动小说的 AI 写手。用户使用 /k 命令，要求你自行决定下一幕的场景并提供开头剧情。'
              f'\n世界观：\n{wbg}{outline_block}\n\n当前幕：{sn}\n'
              f'\n[已有角色档案]{char_profiles}\n'
              '\n请输出一个 JSON 对象（不要 markdown），包含：'
              '\n- premise: 中文描述本幕的地点、时间、开场矛盾和故事走向'
              '\n- characters: 从已有角色中选出的登场角色 codename 列表'
              '\n- opening_hint: 一句简短的开场提示（用于写作模式的起始事件）'
              '\n如果无需指定角色或无可选角色，characters 可以为空列表。')
        msgs = [{"role":"system","content":sp}]
        msgs.append({"role":"user","content":"/k 请自行决定本幕场景并开始写作。"})
        pcb.append({"role":"user","content":"/k（导演要求 AI 自由决定场景）"})
        c = get_client(state)
        r, e = call_deepseek(c, state["model"], msgs)
        if e:
            pcb.append({"role":"assistant","content":f"[错误] {e}"})
            return state, pcb, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        # Parse AI's proposal
        premise = "待定场景"; chars = []; opening = ""
        try:
            m = re.search(r'\{.*\}', r.strip(), re.DOTALL)
            if m:
                d = json.loads(m.group(0))
                premise = d.get("premise","待定场景")
                chars = d.get("characters",[]) or []
                opening = d.get("opening_hint","")
        except: pass
        state["planning_history"].append({"role":"user","content":"/k"})
        state["planning_history"].append({"role":"assistant","content":r})
        pcb.append({"role":"assistant","content":r})
        # Validate characters
        vc = [c for c in chars if c in kc] or chars
        state["scene_premise"] = premise
        state["scene_outline"] = premise
        state["active_characters"] = vc
        state["turn_order"] = build_turn_order(vc)
        state["turn_index"] = 0
        state["mode"] = "WRITING"
        state["writing_history"] = []
        state["current_draft"] = ""
        state["pending_user_input"] = ""
        state["pending_turn_input"] = ""
        state["pending_actor"] = ""
        state["pending_new_characters"] = []
        _clear_revision_state(state)
        _clear_undo_stack(state)
        _reset_scene_support_context(state)
        pcb.append({"role":"assistant","content":f"**AI 自行决定场景：**\n场景: {premise}\n角色: {', '.join(vc) if vc else '未指定'}\n\n--- 进入故事模式 ---"})
        wcb = [{"role":"assistant","content":f"**{state['scene_name']}**\n场景: {premise}\n\n请输入 /k 或你的评价来启动第一个事件。{('开头提示：' + opening) if opening else ''}"}]
        st = f"当前幕: {state['scene_name']} | 场景: {premise[:60]}..."
        return state, pcb, "", wcb, writing_mode_label(state), st, gr.update(visible=False), gr.update(visible=True)

    cc = _build_character_context(state, active_only=False)
    outline_block = _outline_context_block(state)
    sp = ('你是用户的小说导演会议伙伴。请用中文和用户讨论下一幕的地点、时间、登场人物和开场矛盾。'
          f'\n世界观：\n{wbg}{outline_block}\n\n当前幕：{sn}\n'
          f'{cc}'
          f'\n最多讨论 {MAX_PLANNING_TURNS} 轮。'
          '\n如果用户输入 /u，这是最高优先级的上帝指令，必须立刻按它修正当前计划。'
          '\n如果用户输入 /k，表示导演让你自行决定场景并直接开始写作。')
    msgs = [{"role":"system","content":sp}]
    msgs.extend(state.get("planning_history",[]))
    msgs.append({"role":"user","content":f"[上帝指令]\n{command[2:].strip()}" if is_god else command})
    pcb.append({"role":"user","content":command})
    c = get_client(state)
    r, e = call_deepseek(c, state["model"], msgs)
    if e:
        pcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, pcb, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    state["planning_history"].append({"role":"user","content":f"[上帝指令]\n{command[2:].strip()}" if is_god else command})
    state["planning_history"].append({"role":"assistant","content":r})
    state["planning_turns"] = state.get("planning_turns",0) + 1
    pcb.append({"role":"assistant","content":r})
    mt = f"导演会议 ({state['planning_turns']}/{MAX_PLANNING_TURNS})"
    if state["planning_turns"] >= MAX_PLANNING_TURNS:
        state, pcb, wcb, mt, st, plan_update, write_update = _magic_transition(state, pcb)
        return state, pcb, "", wcb, mt, st, plan_update, write_update
    return state, pcb, "", gr.update(), mt, gr.update(), gr.update(), gr.update()

def _magic_transition(state, pcb):
    """Extract scene premise & characters from planning chat. Returns (state, pcb, wcb, mode_text, status_bar, plan_panel_visible, write_panel_visible)."""
    if not state.get("planning_history"):
        pcb.append({"role":"assistant","content":"还没有导演会议记录。"})
        return state, pcb, [], "导演会议", "请先讨论这一幕。", gr.update(visible=True), gr.update(visible=False)
    c = get_client(state)
    cc = _build_character_context(state, active_only=False)
    sp = ("分析这段导演会议。只输出一个合法 JSON 对象，不要 markdown。必须包含 keys: "
          "'premise'（中文字符串，概括地点、时间、目标/冲突）和 'characters'（角色 codename 列表）。"
          f'\n\n[可用角色档案]{cc}\n'
          'Example: {"premise":"Late night at the bar.","characters":["elara","shadow_knight"]}')
    msgs = [{"role":"system","content":sp}] + state["planning_history"]
    rr, e = call_deepseek(c, state["model"], msgs, max_tok=300)
    premise = "待定场景"; chars = []
    if not e and rr:
        m = re.search(r'\{.*\}', rr.strip(), re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                premise = d.get("premise","待定场景")
                chars = d.get("characters",[])
            except: premise = rr[:200]
        else: premise = rr[:200]
    kc = state.get("characters",{})
    vc = [c for c in chars if c in kc] or chars
    state["scene_premise"] = premise
    state["scene_outline"] = premise
    state["active_characters"] = vc
    state["turn_order"] = build_turn_order(vc)
    state["turn_index"] = 0
    state["mode"] = "WRITING"
    state["writing_history"] = []
    state["current_draft"] = ""
    state["pending_user_input"] = ""
    state["pending_turn_input"] = ""
    state["pending_actor"] = ""
    state["pending_new_characters"] = []
    _clear_revision_state(state)
    _clear_undo_stack(state)
    _reset_scene_support_context(state)
    pcb.append({"role":"assistant","content":f"**已提取场景：**\n场景: {premise}\n角色: {', '.join(vc) if vc else '未指定'}\n\n--- 进入故事模式 ---"})
    wcb = [{"role":"assistant","content":f"**{state['scene_name']}**\n场景: {premise}\n\n请输入 /k 来让 AI 开始推演第一个事件，或输入你的评价/指引。"}]
    st = f"当前幕: {state['scene_name']} | 场景: {premise[:60]}..."
    return state, pcb, wcb, writing_mode_label(state), st, gr.update(visible=False), gr.update(visible=True)

def handle_start_writing(state, pcb): return _magic_transition(state, pcb)


def handle_reply_chars_change(state, val):
    """Sync reply_chars slider value into state."""
    if val is not None:
        state["reply_char_count"] = int(val)
    return state

def _writing_tail_updates(show_regen=False):
    """Updates for regen button, hidden discard placeholder, and reply length slider."""
    return gr.update(visible=show_regen), gr.update(visible=False), gr.update(visible=True)

def handle_no_outline_start(state):
    """Enter writing mode without scene premise or AI setup; the first user input defines the opening."""
    sdir = current_story_dir(state)
    wbg = read_file(os.path.join(sdir, "World.txt"))
    state["world_bg"] = wbg
    load_characters_from_char_dir(state)
    sn = state.get("scene_name", "scene_01.txt")
    ac = list(state.get("characters", {}).keys())
    state["scene_premise"] = ""
    state["scene_outline"] = ""
    state["active_characters"] = ac
    state["turn_order"] = build_turn_order(ac)
    state["turn_index"] = 0
    state["mode"] = "WRITING"
    state["writing_history"] = []
    state["current_draft"] = ""
    state["pending_user_input"] = ""
    state["pending_turn_input"] = ""
    state["pending_actor"] = ""
    state["pending_new_characters"] = []
    state["planning_history"] = []
    state["planning_turns"] = 0
    _clear_revision_state(state)
    _clear_undo_stack(state)
    _reset_scene_support_context(state)
    chars_str = ", ".join(ac) if ac else "未指定"
    wcb = [{"role":"assistant","content":f"**{sn}**\n无大纲开始。登场角色: {chars_str}。\n\n请直接输入第一段开场/指引，AI 将据此开始推演。"}]
    mt = writing_mode_label(state)
    st = f"当前幕: {sn} | 场景: 无大纲开始"
    return state, wcb, mt, st, gr.update(visible=False), gr.update(visible=True)

def handle_quick_start(state, scene_brief):
    """Simplified planning: user gives brief, AI internally sets up scene, direct to writing."""
    brief = (scene_brief or "").strip()
    if not brief:
        return handle_no_outline_start(state)
    
    sdir = current_story_dir(state)
    # Re-read world and characters from disk every time a scene starts
    wbg = read_file(os.path.join(sdir, "World.txt"))
    if wbg:
        state["world_bg"] = wbg
    load_characters_from_char_dir(state)  # refresh in-memory char dict from disk
    
    c = get_client(state)
    sn = state.get("scene_name", "scene_01.txt")
    cc = _build_character_context(state, active_only=False)
    outline_block = _outline_context_block(state)
    
    sp = (
        "你是小说场景构建器。用户给了场景简述，请你内部完成场景设定。"
        "只输出合法JSON，不要markdown。必须包含："
        "premise（中文字符串，补全地点/时间/目标/冲突）、"
        "outline（中文字符串，3-6个关键剧情节点）、"
        "characters（角色codename数组）。"
        f"\n\n[世界观]\n{wbg}{outline_block}\n\n[已有角色]\n{cc}\n\n[用户给的场景简述]\n{brief}"
    )
    msgs = [{"role":"system", "content": sp}, {"role":"user", "content": brief}]
    r, e = call_deepseek(c, state["model"], msgs, max_tok=MAX_TOKENS_SCENE_SETUP)
    
    premise = brief[:200]
    outline = ""
    chars = []
    if not e and r:
        try:
            m = re.search(r'\{.*\}', r.strip(), re.DOTALL)
            if m:
                d = json.loads(m.group(0))
                premise = d.get("premise", brief[:200])
                outline = d.get("outline", "")
                chars = d.get("characters", []) or []
        except:
            pass
    
    kc = state.get("characters", {})
    vc = [c for c in chars if c in kc] or list(kc.keys())[:4]
    state["scene_premise"] = premise
    state["scene_outline"] = outline or premise
    state["active_characters"] = vc
    state["turn_order"] = build_turn_order(vc)
    state["turn_index"] = 0
    state["mode"] = "WRITING"
    state["writing_history"] = []
    state["current_draft"] = ""
    state["pending_user_input"] = ""
    state["pending_turn_input"] = ""
    state["pending_actor"] = ""
    state["pending_new_characters"] = []
    state["planning_history"] = []
    state["planning_turns"] = 0
    _clear_revision_state(state)
    _clear_undo_stack(state)
    _reset_scene_support_context(state)
    
    chars_str = ", ".join(vc) if vc else "全部角色"
    wcb = [{"role":"assistant", "content": f"**{sn}**\n场景: {premise}\n\n登场: {chars_str}\n\n请输入评价/指引来启动第一个事件，或输入 /k 让 AI 自行开场。"}]
    mt = writing_mode_label(state)
    st = f"当前幕: {sn} | 场景: {premise[:60]}..."
    return state, wcb, mt, st, gr.update(visible=False), gr.update(visible=True)

def handle_skip_planning(state, pcb, mp):
    premise = mp.strip() if mp else "待定场景"
    ac = list(state.get("characters",{}).keys())
    state.update({"scene_premise":premise,"active_characters":ac,"mode":"WRITING",
        "turn_order":build_turn_order(ac),"turn_index":0,
        "writing_history":[],"current_draft":"","pending_user_input":"","scene_outline":premise,
        "pending_turn_input":"","pending_actor":"","pending_new_characters":[]})
    _clear_revision_state(state)
    _clear_undo_stack(state)
    _reset_scene_support_context(state)
    pcb.append({"role":"assistant","content":f"已跳过导演会议。\n场景: {premise}"})
    wcb = [{"role":"assistant","content":f"**{state['scene_name']}**\n场景: {premise}\n\n请输入 /k 来让 AI 开始推演第一个事件，或输入你的评价/指引。"}]
    mt = writing_mode_label(state)
    st = f"当前幕: {state['scene_name']} | 场景: {premise[:60]}..."
    return state, pcb, wcb, mt, st, gr.update(visible=False), gr.update(visible=True)

# ---- Shared Helper: build character context string from state ----
def _context_character_codenames(state, active_only=True):
    sdir = current_story_dir(state)
    chars = state.get("characters", {}) or {}
    ac = state.get("active_characters", []) or []
    if active_only and ac:
        return list(ac)
    target_codenames = list(ac) if ac else sorted(chars.keys())
    if not target_codenames:
        cdir = os.path.join(sdir, "char")
        if os.path.isdir(cdir):
            target_codenames = sorted(
                fn[:-4] for fn in os.listdir(cdir)
                if fn.endswith(".txt") and not fn.startswith(".")
            )
    return target_codenames

def _build_character_context(state, active_only=True):
    """Build a unified character context string from state.
    
    If active_only=True, uses active_characters only.
    If active_only=False or active_characters is empty, falls back to all characters.
    Always reads actual .txt files if the in-memory dict has empty values.
    """
    sdir = current_story_dir(state)
    chars = state.get("characters", {})
    target_codenames = _context_character_codenames(state, active_only=active_only)
    
    if not target_codenames:
        return ""
    
    cc_parts = []
    for cn in target_codenames:
        # Always read from disk to get latest version (user may have edited via build tab)
        p = read_file(character_path(sdir, cn))
        if not p:
            p = chars.get(cn, "")  # fallback to in-memory
        if p:
            cc_parts.append(f"\n--- {cn} ---\n{p}\n")
            state["characters"][cn] = p  # sync memory
        else:
            cc_parts.append(f"\n--- {cn} ---\n（角色档案未找到）\n")
    
    return "".join(cc_parts)

def _build_character_personality_context(state, active_only=True):
    sdir = current_story_dir(state)
    chars = state.get("characters", {}) or {}
    lines = []
    for cn in _context_character_codenames(state, active_only=active_only):
        profile = read_file(character_path(sdir, cn)) or chars.get(cn, "")
        if not profile:
            continue
        name = extract_character_name(profile) or cn
        fields = []
        for field in ["性格特征", "爱好", "人际关系", "能力", "过往经历"]:
            value = _character_field(profile, field)
            if value:
                fields.append(f"{field}：{limit_chars(value, 220)}")
        if fields:
            lines.append(f"{name}（{cn}）：{'；'.join(fields)}")
    return "\n".join(lines)

def _story_section_message(identifier, title, content, role="system"):
    content = str(content or "").strip()
    if not content:
        return None
    return {"role": role, "content": f"[{title}]\n{content}", "_id": identifier}

def _clean_story_messages(messages):
    cleaned = []
    for msg in messages:
        if not msg:
            continue
        item = {k: v for k, v in msg.items() if not k.startswith("_")}
        if str(item.get("content", "")).strip():
            cleaned.append(item)
    return cleaned

def _story_world_info_before(state):
    parts = []
    if state.get("world_bg"):
        parts.append(f"[World.txt]\n{state.get('world_bg')}")
    past_outline = _past_outline(state)
    if past_outline:
        parts.append(f"[outline.txt]\n{past_outline}")
    return "\n\n".join(parts)

def _story_world_info_after(state):
    parts = []
    fact_text = _fact_pack_text(state)
    if fact_text:
        parts.append(f"[事实包]\n{fact_text}")
    if state.get("scene_summary"):
        parts.append(f"[当前场景总结]\n{state.get('scene_summary')}")
    return "\n\n".join(parts)

def _story_scenario_context(state):
    parts = []
    sp = state.get("scene_premise", "")
    outline = state.get("scene_outline", "")
    if sp:
        parts.append(sp)
    if outline and outline.strip() != sp.strip():
        parts.append(f"[场景构架]\n{outline}")
    if not parts:
        parts.append("无大纲开始：以导演当前输入作为开场锚点，不补造未给出的过去场景。")
    return "\n\n".join(parts)

def _story_main_prompt(purpose="event"):
    if purpose == "god":
        return (
            "你是互动小说的最高优先级修正器。用户会用 /u 发出上帝指令，用来修正剧情错误、连续性问题、角色误写或世界观冲突。"
            "必须服从上帝指令，并输出可直接写入正式故事记录的修正版剧情/补丁。"
        )
    if purpose == "synopsis":
        return "你是互动小说写手。导演使用 /s 命令，需要你从当前剧情点出发，用大纲完成当前场景。"
    return "你是互动小说写手。用户是故事导演；请依据正式历史、当前设定和导演输入，用中文继续当前事件。"

def _story_control_prompt(state, purpose="event"):
    chars_count = state.get("reply_char_count", DEFAULT_REPLY_CHARS)
    common = [
        "只使用正式历史、当前设定、事实包和导演本轮输入；候选草稿只在修订链中作为上一版参考。",
        "过去章节提纲只用于连续性，禁止复述或重演旧章节。",
        "保持角色年龄、身份、关系、能力、知识范围和当前状态一致。",
        "只输出正文内容，不写戏外解释、元评论、系统规则或导演原话。",
    ]
    if purpose == "god":
        rules = [
            "上帝指令优先级最高；按指令修正剧情错误。",
            "不要消耗任何角色回合，不要让当前角色额外行动，除非上帝指令明确要求。",
            f"输出控制在 {chars_count} 个中文字符以内，除非修正内容必须更长。",
        ]
    elif purpose == "synopsis":
        rules = [
            "输出完整场景完成大纲，一直列到自然收束。",
            "使用条目或短横线列表（8-14条），不要写成小说段落。",
            "每条尽量包含触发事件、关键行动、NPC/环境反应、结果或代价。",
            "最后一条写清楚场景如何收束，以及留下的悬念、奖励或后果。",
        ]
    else:
        rules = [
            f"根据 UI 回复字数上限推进剧情，不超过 {chars_count} 个中文字符。",
            "聚焦当前行动、对话、环境变化和角色反应。",
            "如果导演给出长期目标，本段只写下一个触发、阻碍或进展；明确要求立刻完成时除外。",
        ]
    return "\n".join(f"- {item}" for item in rules + common)

def _story_context_messages(state, purpose="event", include_control=False):
    cc = _build_character_context(state, active_only=True)
    if not cc:
        cc = _build_character_context(state, active_only=False)
    cp = _build_character_personality_context(state, active_only=True)
    if not cp:
        cp = _build_character_personality_context(state, active_only=False)
    messages = [
        _story_section_message("worldInfoBefore", "世界信息 / 过去提纲", _story_world_info_before(state)),
        _story_section_message("main", "主提示", _story_main_prompt(purpose)),
        _story_section_message("worldInfoAfter", "当前记忆 / 事实约束", _story_world_info_after(state)),
        _story_section_message("charDescription", "角色卡", cc.strip() if cc else ""),
        _story_section_message("charPersonality", "角色性格 / 行为准则", cp),
        _story_section_message("scenario", "本幕场景", _story_scenario_context(state)),
        _story_section_message("personaDescription", "导演身份", "用户是故事导演。用户输入是创作指令，不是角色台词，除非用户明确要求写入正文。"),
    ]
    if include_control:
        messages.append(_story_section_message("controlPrompts", "输出约束", _story_control_prompt(state, purpose)))
    return _clean_story_messages(messages)

def _story_control_message(state, purpose="event"):
    return _clean_story_messages([_story_section_message("controlPrompts", "输出约束", _story_control_prompt(state, purpose))])[0]

def _build_story_api_messages(state, user_prompt, purpose="event", revision_history=None):
    msgs = _story_context_messages(state, purpose=purpose)
    msgs.extend(_api_writing_context(state))
    if revision_history:
        msgs.extend([dict(m) for m in revision_history])
    msgs.append(_story_control_message(state, purpose=purpose))
    if user_prompt is not None:
        msgs.append({"role": "user", "content": str(user_prompt)})
    return msgs


# ---- Writing Handlers ----
def _build_wsp(state):
    return "\n\n".join(m["content"] for m in _story_context_messages(state, purpose="event", include_control=True))

def _build_event_prompt(state):
    """Build the user prompt for requesting the next event."""
    chars_count = state.get("reply_char_count", DEFAULT_REPLY_CHARS)
    return f"自行推演后续剧情，控制在 {chars_count} 个中文字符以内。"

CONTEXT_WARN_TOKENS = 500000  # 50W token warning threshold

def _estimate_context_tokens(state):
    """Roughly estimate total tokens in the current scene context (writing_history)."""
    wh = _api_writing_context(state)
    rejected = state.get("rejected_draft_history", [])
    total_chars = sum(len(m.get("content", "")) for m in wh + rejected)
    # Also count system prompt overhead
    system_chars = len(state.get("world_bg", "")) + len(state.get("scene_premise", ""))
    total_chars += system_chars
    return int(total_chars * TOKEN_MULTIPLIER_PER_CHAR)

def _check_context_size(state, wcb):
    """Warn user if context is approaching the limit."""
    estimated = _estimate_context_tokens(state)
    if estimated > CONTEXT_WARN_TOKENS:
        wcb.append({"role": "assistant",
            "content": f"⚠️ **上下文警告**：当前幕已累积约 {estimated // 1000}K tokens（上限 {CONTEXT_WARN_TOKENS // 1000}K），"
                       f"建议尽快「结束本幕」以重置上下文，避免超出 AI 处理上限。"})
        return True
    elif estimated > CONTEXT_WARN_TOKENS * 0.7:
        wcb.append({"role": "assistant",
            "content": f"⚡ **上下文提醒**：当前幕已累积约 {estimated // 1000}K tokens，接近 {CONTEXT_WARN_TOKENS // 1000}K 上限。"
                       f"建议在适当时候「结束本幕」。**（本条提醒仅显示一次，之后不再重复）**"})
        return True
    return False

def _build_skip_synopsis_prompt(state):
    """Build system prompt for /s: AI generates a full scene outline."""
    return "\n\n".join(m["content"] for m in _story_context_messages(state, purpose="synopsis", include_control=True))

def _build_god_command_prompt(state):
    prompt = "\n\n".join(m["content"] for m in _story_context_messages(state, purpose="god", include_control=True))
    return f"{prompt}\n\n[当前回合]\n{turn_label(state)}"

SETTING_UPDATE_ACTORS = {"__char_update__", "__world_update__"}

def _setting_update_label(kind):
    return "角色卡" if kind == "character" else "世界设定"

def _command_text(command, prefix):
    return (command or "")[len(prefix):].strip()

def _capture_story_draft_state(state):
    return {
        "current_draft": state.get("current_draft", ""),
        "pending_user_input": state.get("pending_user_input", ""),
        "pending_turn_input": state.get("pending_turn_input", ""),
        "pending_actor": state.get("pending_actor", ""),
        "rejected_draft_history": [dict(m) for m in state.get("rejected_draft_history", [])],
        "revision_ui_start": state.get("revision_ui_start"),
    }

def _restore_story_draft_state(state, snapshot):
    snapshot = snapshot or {}
    state["current_draft"] = snapshot.get("current_draft", "")
    state["pending_user_input"] = snapshot.get("pending_user_input", "")
    state["pending_turn_input"] = snapshot.get("pending_turn_input", "")
    state["pending_actor"] = snapshot.get("pending_actor", "")
    state["rejected_draft_history"] = [dict(m) for m in snapshot.get("rejected_draft_history", [])]
    state["revision_ui_start"] = snapshot.get("revision_ui_start")
    state["pending_setting_update"] = None
    return state

def _setting_controls_visible(state):
    return bool(state.get("current_draft"))

def _remove_setting_ui(state, wcb):
    pending = state.get("pending_setting_update") or {}
    start = pending.get("ui_start")
    if isinstance(start, int) and 0 <= start <= len(wcb):
        del wcb[start:]
    return pending

def _clear_existing_setting_ui(state, wcb):
    pending = _remove_setting_ui(state, wcb)
    _restore_story_draft_state(state, pending.get("story_snapshot", {}))

def _prepare_setting_layer(state, wcb):
    if state.get("pending_actor") in SETTING_UPDATE_ACTORS:
        pending = _remove_setting_ui(state, wcb)
        snapshot = pending.get("story_snapshot", {})
        _restore_story_draft_state(state, snapshot)
        return snapshot
    return _capture_story_draft_state(state)

def _cancel_setting_update(state, wcb):
    pending = _remove_setting_ui(state, wcb)
    _restore_story_draft_state(state, pending.get("story_snapshot", {}))
    return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(_setting_controls_visible(state))

def _recent_story_context_text(state):
    recent = _recent_writing_history_window(state)
    if not recent:
        return "无"
    lines = []
    for msg in recent:
        label = "导演" if msg.get("role") == "user" else "剧情"
        lines.append(f"{label}: {msg.get('content','')}")
    return "\n".join(lines)

def _build_character_update_prompt(state):
    chars = state.get("characters", {}) or {}
    char_context = _build_character_context(state, active_only=False)
    active = ", ".join(state.get("active_characters", []) or []) or "未指定"
    return (
        "你是互动小说的角色卡维护器。用户会用 /c 在故事进行中微调或创建角色设定。\n"
        "你必须根据用户要求，更新已有角色卡，或创建新的完整角色卡。不要推进剧情，不要写小说正文。\n"
        "只输出合法 JSON 对象，不要 markdown，不要解释。格式：\n"
        '{"summary":"会如何处理角色卡的简述","updates":[{"action":"update/create","codename":"角色ID","name":"角色名","reason":"为什么处理","profile":"更新后或新建的完整角色卡"}]}\n'
        "规则：\n"
        "- action 为 update 时，codename 必须来自已有角色ID。\n"
        "- action 为 create 时，codename 必须是新的英文/拼音/数字/下划线 ID，不得与已有角色ID重复。\n"
        "- profile 必须是更新后或新建后的完整角色卡，不是补丁片段。\n"
        "- 每张角色卡必须保持字段和顺序：姓名、性别、年龄、职业、外貌、性格特征、爱好、过往经历、人际关系、能力。\n"
        "- 姓名行括号中的 ID 必须与 codename 完全一致。\n"
        "- 只处理用户要求影响到的角色；不要主动建议无关新角色。\n"
        f"\n[已有角色ID]\n{', '.join(sorted(chars.keys())) if chars else '无'}"
        f"\n\n[当前登场角色]\n{active}"
        f"\n\n[世界观]\n{state.get('world_bg','')}"
        f"\n\n[本幕场景]\n{state.get('scene_premise','')}"
        f"\n\n[最近正式剧情]\n{_recent_story_context_text(state)}"
        f"\n\n[全部角色卡]\n{char_context if char_context else '无'}"
    )

def _build_world_update_prompt(state):
    return (
        "你是互动小说的世界设定维护器。用户会用 /w 在故事进行中微调世界设定。\n"
        "你必须根据用户要求，给出确认后可直接写入 World.txt 的完整世界设定。不要推进剧情，不要写小说正文。\n"
        "只输出合法 JSON 对象，不要 markdown，不要解释。格式：\n"
        '{"summary":"会如何更新世界设定的简述","updated_world":"更新后的完整 World.txt 内容"}\n'
        "规则：\n"
        "- updated_world 必须保留原世界观中仍然有效的重要信息。\n"
        "- 如果用户是在修正矛盾，直接在 updated_world 中改正矛盾，不要只追加备注。\n"
        "- 内容尽量精炼，但不要删除维持连续性所需的信息。\n"
        f"\n[当前 World.txt]\n{state.get('world_bg','') or '空'}"
        f"\n\n[本幕场景]\n{state.get('scene_premise','')}"
        f"\n\n[最近正式剧情]\n{_recent_story_context_text(state)}"
    )

def _unique_new_character_codename(state, base, reserved=None):
    sdir = current_story_dir(state)
    reserved = set(reserved or set())
    existing = set((state.get("characters", {}) or {}).keys()) | reserved
    base = sanitize_codename(base or "new_character")
    candidate = unique_character_codename(sdir, base)
    if candidate not in existing:
        return candidate
    index = 1
    while True:
        candidate = unique_character_codename(sdir, f"{base}_{index}")
        if candidate not in existing:
            return candidate
        index += 1

def _valid_character_updates(state, data):
    updates = data.get("updates", []) if isinstance(data, dict) else []
    if not isinstance(updates, list):
        return []
    existing = set((state.get("characters", {}) or {}).keys())
    used = set()
    valid = []
    for item in updates:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().lower()
        name = str(item.get("name", "")).strip()
        raw_cn = str(item.get("codename", "")).strip()
        cn = sanitize_codename(raw_cn) if raw_cn else ""
        profile = str(item.get("profile", "")).strip()
        if not profile:
            continue
        if not cn:
            cn = local_codename_from_text(name or profile, fallback="new_character")
        if action not in ("update", "create"):
            action = "update" if cn in existing else "create"
        if action == "update" and cn not in existing:
            action = "create"
        if action == "create":
            cn = _unique_new_character_codename(state, cn, used)
        if cn in used:
            cn = _unique_new_character_codename(state, cn, used)
            action = "create"
        used.add(cn)
        fallback_name = (
            extract_character_name(state.get("characters", {}).get(cn, ""))
            if action == "update" else ""
        ) or name or extract_character_name(profile) or cn
        profile = normalize_character_profile(profile, cn, fallback_name)
        valid.append({
            "action": action,
            "codename": cn,
            "name": fallback_name,
            "reason": str(item.get("reason", "")).strip(),
            "profile": profile,
        })
    return valid

def _format_character_update_proposal(state, data, raw):
    summary = str(data.get("summary", "")).strip() if isinstance(data, dict) else ""
    updates = _valid_character_updates(state, data)
    if not updates:
        return f"*角色修改*：\n*角色卡草稿解析失败，请重新生成或重新输入 /c。*\n\n{raw}"
    parts = ["*角色修改*：", "*角色卡创建/更新草稿（确认后写入角色卡并重新载入；不写入剧情记录）：*"]
    if summary:
        parts.append(f"摘要：{summary}")
    for item in updates:
        cn = item["codename"]
        profile = item["profile"]
        label = "新建" if item.get("action") == "create" else "更新"
        reason = f"\n原因：{item['reason']}" if item.get("reason") else ""
        parts.append(f"\n--- {label} {cn} ---{reason}\n{profile}")
    data["updates"] = updates
    return "\n".join(parts)

def _format_world_update_proposal(data, raw):
    summary = str(data.get("summary", "")).strip() if isinstance(data, dict) else ""
    updated_world = str(data.get("updated_world", "") or data.get("world_bg", "") or data.get("world", "")).strip() if isinstance(data, dict) else ""
    if not updated_world:
        return f"*世界修改*：\n*世界设定更新草稿解析失败，请重新生成或重新输入 /w。*\n\n{raw}"
    data["updated_world"] = updated_world
    parts = ["*世界修改*：", "*世界设定更新草稿（确认后写入 World.txt 并重新载入；不写入剧情记录）：*"]
    if summary:
        parts.append(f"摘要：{summary}")
    parts.append(f"\n{updated_world}")
    return "\n".join(parts)

def _setting_update_prompt(state, kind):
    return _build_character_update_prompt(state) if kind == "character" else _build_world_update_prompt(state)

def _format_setting_update_proposal(state, kind, data, raw):
    return _format_character_update_proposal(state, data, raw) if kind == "character" else _format_world_update_proposal(data, raw)

def _start_setting_update(state, command, wcb, kind):
    prefix = "/c" if kind == "character" else "/w"
    instruction = _command_text(command, prefix)
    label = _setting_update_label(kind)
    if not instruction:
        wcb.append({"role":"assistant","content":f"请在 {prefix} 后面写明要更新的{label}内容。"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
    if kind == "character":
        load_characters_from_char_dir(state)
    story_snapshot = _prepare_setting_layer(state, wcb)
    ui_start = len(wcb)
    state["pending_user_input"] = command
    state["pending_turn_input"] = instruction
    state["pending_actor"] = "__char_update__" if kind == "character" else "__world_update__"
    state["pending_setting_update"] = {
        "type": kind,
        "ui_start": ui_start,
        "story_snapshot": story_snapshot,
        "instruction": instruction,
        "messages": [{"role": "user", "content": instruction}],
        "data": {},
        "raw": "",
    }
    msgs = [{"role":"system","content":_setting_update_prompt(state, kind)}, {"role":"user","content":instruction}]
    wcb.append({"role":"user","content":command})
    c = get_client(state)
    max_tok = 4000 if kind == "character" else 1800
    raw, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok)
    if e:
        wcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)
    data = extract_json_object(raw)
    display = _format_setting_update_proposal(state, kind, data, raw)
    state["pending_setting_update"].update({
        "messages": [{"role":"user","content":instruction}, {"role":"assistant","content":raw}],
        "data": data,
        "raw": raw,
    })
    state["current_draft"] = display
    wcb.append({"role":"assistant","content":display})
    return state, wcb, "", f"故事模式 | {label}更新待确认", gr.update(), *_writing_tail_updates(True)

def _continue_setting_update(state, command, wcb):
    pending = state.get("pending_setting_update") or {}
    kind = pending.get("type")
    if kind not in ("character", "world"):
        _clear_pending_draft(state)
        return handle_writing_send(state, command, wcb)
    instruction = (command or "").strip()
    if not instruction:
        wcb.append({"role":"assistant","content":"请写明要如何调整这次设定更新。"})
        return state, wcb, "", f"故事模式 | {_setting_update_label(kind)}更新待确认", gr.update(), *_writing_tail_updates(True)
    pending.setdefault("messages", []).append({"role":"user","content":instruction})
    msgs = [{"role":"system","content":_setting_update_prompt(state, kind)}] + pending.get("messages", [])
    wcb.append({"role":"user","content":command})
    c = get_client(state)
    max_tok = 4000 if kind == "character" else 1800
    raw, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok)
    if e:
        wcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, wcb, "", f"故事模式 | {_setting_update_label(kind)}更新待确认", gr.update(), *_writing_tail_updates(True)
    data = extract_json_object(raw)
    display = _format_setting_update_proposal(state, kind, data, raw)
    pending["messages"].append({"role":"assistant","content":raw})
    pending.update({"data": data, "raw": raw})
    state["pending_setting_update"] = pending
    state["current_draft"] = display
    wcb.append({"role":"assistant","content":display})
    return state, wcb, "", f"故事模式 | {_setting_update_label(kind)}更新待确认", gr.update(), *_writing_tail_updates(True)

def _apply_setting_update(state, wcb):
    pending = state.get("pending_setting_update") or {}
    kind = pending.get("type")
    data = pending.get("data") if isinstance(pending.get("data"), dict) else extract_json_object(pending.get("raw", ""))
    sdir = current_story_dir(state)
    if kind == "character":
        updates = _valid_character_updates(state, data)
        if not updates:
            wcb.append({"role":"assistant","content":"*无法应用角色卡草稿：没有解析到有效角色 ID 和完整角色卡。*"})
            return state, wcb, "", "故事模式 | 角色卡更新待确认", gr.update(), *_writing_tail_updates(True)
        created = []
        updated = []
        for item in updates:
            cn = item["codename"]
            action = item.get("action", "update")
            fallback_name = item.get("name") or extract_character_name(state.get("characters", {}).get(cn, "")) or cn
            profile = normalize_character_profile(item["profile"], cn, fallback_name)
            write_file(character_path(sdir, cn), profile)
            if action == "create":
                created.append(cn)
            else:
                updated.append(cn)
        load_characters_from_char_dir(state)
        status_parts = []
        if created:
            status_parts.append(f"已新建角色卡：{', '.join(created)}")
        if updated:
            status_parts.append(f"已更新角色卡：{', '.join(updated)}")
        status = "故事模式 | " + "；".join(status_parts)
    elif kind == "world":
        updated_world = str(data.get("updated_world", "") or data.get("world_bg", "") or data.get("world", "")).strip() if isinstance(data, dict) else ""
        if not updated_world:
            wcb.append({"role":"assistant","content":"*无法应用世界设定更新：没有解析到 updated_world。*"})
            return state, wcb, "", "故事模式 | 世界设定更新待确认", gr.update(), *_writing_tail_updates(True)
        write_file(os.path.join(sdir, "World.txt"), updated_world)
        state["world_bg"] = read_file(os.path.join(sdir, "World.txt"))
        status = "故事模式 | 已更新世界设定"
    else:
        wcb.append({"role":"assistant","content":"*当前没有可确认的设定更新。*"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)

    start = pending.get("ui_start")
    if isinstance(start, int) and 0 <= start <= len(wcb):
        del wcb[start:]
    _restore_story_draft_state(state, pending.get("story_snapshot", {}))
    _refresh_fact_pack(state, get_client(state), force=True)
    return state, wcb, "", status, gr.update(), *_writing_tail_updates(_setting_controls_visible(state))

def _regenerate_setting_update(state, wcb):
    pending = state.get("pending_setting_update") or {}
    kind = pending.get("type")
    if kind not in ("character", "world"):
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
    messages = [dict(m) for m in pending.get("messages", [])]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    if not messages:
        messages = [{"role":"user","content":pending.get("instruction", "")}]
    msgs = [{"role":"system","content":_setting_update_prompt(state, kind)}] + messages
    if wcb and wcb[-1].get("role") == "assistant":
        wcb.pop()
    c = get_client(state)
    max_tok = 4000 if kind == "character" else 1800
    raw, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok)
    if e:
        wcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, wcb, "", f"故事模式 | {_setting_update_label(kind)}更新待确认", gr.update(), *_writing_tail_updates(True)
    data = extract_json_object(raw)
    display = _format_setting_update_proposal(state, kind, data, raw)
    pending.update({
        "messages": messages + [{"role":"assistant","content":raw}],
        "data": data,
        "raw": raw,
    })
    state["pending_setting_update"] = pending
    state["current_draft"] = display
    wcb.append({"role":"assistant","content":display})
    return state, wcb, "", f"故事模式 | {_setting_update_label(kind)}更新待确认", gr.update(), *_writing_tail_updates(True)

def _compress_character_if_needed(state, cn):
    """If character profile exceeds 1000 chars, ask AI to compress Memory Updates into 过往经历."""
    sdir = current_story_dir(state)
    fp = character_path(sdir, cn)
    profile = read_file(fp)
    if not profile or len(profile) <= 1000:
        return False
    c = get_client(state)
    sp = (
        "你是角色档案压缩器。角色档案包含固定字段和多次追加的 [Memory Update] 记忆块。\n"
        "请保持字段结构不变，但将「过往经历」字段与所有 [Memory Update] 内容融合，"
        "改写成 1-2 段流畅的中文叙述，概括角色迄今最重要的经历和变化。\n"
        "严格控制在 500 中文字以内。其他字段（姓名/性别/年龄/职业/外貌/性格特征/爱好/人际关系/能力）保持不变。\n"
        "只输出完整角色档案，不要解释。"
    )
    raw, err = call_deepseek(c, state["model"], [
        {"role": "system", "content": sp},
        {"role": "user", "content": f"请压缩以下角色档案：\n\n{profile}"}
    ], max_tok=MAX_TOKENS_MEMORY_COMPRESSION)
    if err or not raw:
        return False
    compressed = normalize_character_profile(raw.strip(), cn)
    compressed = limit_words(compressed)
    if compressed:
        write_file(fp, compressed)
        state["characters"][cn] = compressed
        return True
    return False

def _settle_scene(state):
    history = state.get("writing_history",[])
    state["_last_scene_summary"] = ""
    state["pending_new_characters"] = []
    if not history:
        return "本幕没有已保存剧情，已跳过结算。"

    sdir = current_story_dir(state)
    history_text = "\n".join([
        f"[{'SYSTEM/PLAYER' if msg['role'] == 'user' else 'AI/STORY'}]: {msg['content']}"
        for msg in history
    ])
    character_contexts = _build_character_context(state, active_only=True)
    if not character_contexts:
        character_contexts = _build_character_context(state, active_only=False)

    existing = sorted(state.get("characters",{}).keys())
    sp = (
        "你是互动小说的连续性编辑。在一幕结束时，你要整理剧情，并判断哪些持久化文件需要更新。\n"
        "只输出合法 JSON 对象，不要 markdown。必须包含这些 key：\n"
        "- scene_summary: 用中文简要总结本幕发生了什么。\n"
        "- world_update: 用中文写重要的新世界观、地点、势力、规则变化；如果没有，写 \"NONE\"。\n"
        "- character_memory_updates: 对象，key 是现有登场角色 codename，value 是 1-3 句中文记忆更新；如果没有，写 \"NONE\"。\n"
        "不要建议或创建新角色；故事模式的新角色只能由导演使用 /c 单独创建。"
    )
    up = (
        f"[世界观]\n{state.get('world_bg','')}\n\n"
        f"[现有角色 codename]\n{', '.join(existing) if existing else '无'}\n\n"
        f"[登场角色档案]\n{character_contexts if character_contexts else '无'}\n\n"
        f"[当前场景总结]\n{state.get('scene_summary','') or '无'}\n\n"
        f"[本幕记录]\n{history_text}"
    )
    c = get_client(state)
    raw, err = call_deepseek(c, state["model"], [{"role":"system","content":sp},{"role":"user","content":up}], max_tok=2500)
    if err:
        return f"结算失败：{err}"

    settlement = extract_json_object(raw)
    if not settlement:
        return f"结算结果无法解析：{raw[:300]}"

    notes = []
    scene_summary = str(settlement.get("scene_summary","")).strip()
    if scene_summary:
        state["_last_scene_summary"] = scene_summary
        append_file(state.get("scene_path",""), f"[Scene Summary]\n{scene_summary}")
        notes.append("已保存本幕摘要。")

    mems = settlement.get("character_memory_updates",{})
    if isinstance(mems, dict):
        for cn, mem in mems.items():
            if cn not in state.get("active_characters",[]):
                continue
            mem = str(mem).strip()
            if not mem or mem.upper() == "NONE":
                continue
            fp = character_path(sdir, cn)
            if os.path.exists(fp):
                append_file(fp, f"[Memory Update]\n{mem}")
                state["characters"][cn] = read_file(fp)
                notes.append(f"已更新 {cn} 的记忆。")
                if _compress_character_if_needed(state, cn):
                    notes.append(f"已压缩 {cn} 的角色档案。")

    world_update = str(settlement.get("world_update","")).strip()
    if world_update and world_update.upper() != "NONE":
        state["world_bg"] = (state.get("world_bg","").strip() + f"\n\n[World Update]\n{world_update}").strip()
        write_file(os.path.join(sdir,"World.txt"), state["world_bg"])
        notes.append("已更新世界观。")

    if not notes:
        notes.append("本幕没有需要写入的角色记忆或世界观更新。")

    return "\n".join(notes)

def handle_writing_send(state, ui, wcb):
    command = (ui or "").strip()
    lower_command = command.lower()
    is_k = lower_command == "/k"

    if lower_command.startswith("/c"):
        return _start_setting_update(state, command, wcb, "character")

    if lower_command.startswith("/w"):
        return _start_setting_update(state, command, wcb, "world")

    # Handle /u god command (highest priority)
    if lower_command.startswith("/u"):
        if state.get("pending_actor") in SETTING_UPDATE_ACTORS:
            _clear_existing_setting_ui(state, wcb)
        instruction = command[2:].strip()
        if not instruction:
            wcb.append({"role":"assistant","content":"请在 /u 后面写明要修正什么。"})
            return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
        if state.get("current_draft"):
            _remember_rejected_current_draft(state, wcb, remove_from_ui=True)
        state["pending_user_input"] = command
        state["pending_turn_input"] = f"[上帝指令]\n{instruction}"
        state["pending_actor"] = "__god__"
        c = get_client(state)
        _refresh_scene_support_context(state, c)
        revision_history = list(state.get("rejected_draft_history", []))
        user_msg = {"role":"user","content":f"[上帝指令]\n{instruction}"}
        msgs = _build_story_api_messages(state, user_msg["content"], purpose="god", revision_history=revision_history)
        wcb.append({"role":"user","content":command})
        max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS), is_director=True)
        draft, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok)
        if e:
            wcb.append({"role":"assistant","content":f"[错误] {e}"})
            return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)
        draft = _validate_and_rewrite_draft(state, c, draft, msgs, max_tok)
        if revision_history:
            state["rejected_draft_history"] = revision_history + [user_msg]
        state["current_draft"] = draft
        wcb.append({"role":"assistant","content":f"*上帝指令修正草稿：*\n\n{draft}"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)

    if state.get("pending_actor") in SETTING_UPDATE_ACTORS:
        if is_k or lower_command.startswith("/s"):
            return _cancel_setting_update(state, wcb)
        return _continue_setting_update(state, command, wcb)

    # Handle /s skip-to-synopsis: AI generates full scene outline
    if lower_command.startswith("/s"):
        # Extract optional director comment after /s
        s_comment = command[2:].strip()
        # Save pending draft first
        if state.get("current_draft"):
            commit_info = _commit_current_draft(state, wcb)
            _replace_revision_ui_with_accepted(wcb, commit_info)
            if not (commit_info and commit_info.get("had_revision")):
                wcb.append({"role":"assistant","content":"*事件已保存。导演使用 /s，AI 用大纲完成当前场景...*"})
        if s_comment:
            user_prompt = f"[导演对大纲的评论]\n{s_comment}\n\n请根据以上评论重新生成当前场景的完整大纲，从当前剧情点列到本场景收束。使用条目/编号。"
            user_input_label = f"/s {s_comment}"
        else:
            user_prompt = "请生成当前场景的完整大纲，从当前剧情点列到本场景收束。使用条目/编号，不要写小说段落。"
            user_input_label = "/s（导演跳过细节，用大纲完成场景）"
        revision_history = list(state.get("rejected_draft_history", []))
        if revision_history:
            feedback_msg = _revision_feedback_message(s_comment)
            user_prompt = feedback_msg["content"] + "\n\n随后请用大纲形式完成当前场景。"
            user_input_label = f"/s {s_comment}".strip() or "/s（废弃后用大纲重写）"
        state["pending_user_input"] = user_input_label
        state["pending_turn_input"] = user_prompt
        state["pending_actor"] = "__skip__"
        c = get_client(state)
        _refresh_scene_support_context(state, c)
        user_msg = {"role":"user","content":user_prompt}
        msgs = _build_story_api_messages(state, user_msg["content"], purpose="synopsis", revision_history=revision_history)
        wcb.append({"role":"user","content":user_input_label})
        _check_context_size(state, wcb)
        max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS) * 5, is_director=True)
        draft, e = call_deepseek(c, state["model"], msgs, max_tok=max(max_tok, MAX_TOKENS_SCENE_SYNOPSIS))
        if e:
            wcb.append({"role":"assistant","content":f"[错误] {e}"})
            return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)
        if revision_history:
            state["rejected_draft_history"] = revision_history + [user_msg]
        state["current_draft"] = draft
        wcb.append({"role":"assistant","content":f"*场景完成大纲草稿（可直接保存后结束本幕）：*\n\n{draft}"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)

    # If /k and there's a pending draft, save it first (director passes on this event)
    if is_k and state.get("current_draft"):
        commit_info = _commit_current_draft(state, wcb)
        _replace_revision_ui_with_accepted(wcb, commit_info)
        if not (commit_info and commit_info.get("had_revision")):
            wcb.append({"role":"assistant","content":"*事件已通过。导演使用 /k，AI 继续推演下一事件...*"})

    # If the director types feedback while a draft is pending, treat it as
    # rejecting that draft and revising the same event.
    if not is_k and state.get("current_draft"):
        _remember_rejected_current_draft(state, wcb, remove_from_ui=True)

    # Build the user prompt for this event
    if is_k:
        # /k: director passes on this event, AI freely decides next event
        user_prompt = _build_event_prompt(state)
        user_input_label = "/k（导演通过，AI 自由推演）"
    elif command:
        # Director gives feedback/guidance for this event
        user_prompt = f"[导演指示] 请参考以下意见推演事件：\n{command}"
        user_input_label = command
    else:
        # Empty input: just request next event
        user_prompt = _build_event_prompt(state)
        user_input_label = "（导演请求下一事件）"

    revision_history = list(state.get("rejected_draft_history", []))
    revision_feedback = None
    if revision_history:
        revision_feedback = _revision_feedback_message(command)
        user_prompt = revision_feedback["content"]
        user_input_label = command if command else "（废弃后重写）"
        if is_k:
            user_input_label = "/k（废弃后重写）"

    state["pending_user_input"] = user_input_label
    state["pending_turn_input"] = user_prompt
    state["pending_actor"] = ""

    c = get_client(state)
    _refresh_scene_support_context(state, c)
    user_msg = {"role":"user","content":user_prompt}
    msgs = _build_story_api_messages(state, user_msg["content"], purpose="event", revision_history=revision_history)
    wcb.append({"role":"user","content":user_input_label})
    # Context size check
    _check_context_size(state, wcb)
    max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS), is_director=True)
    draft, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok)
    if e:
        wcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, wcb, "", gr.update(), gr.update(), *_writing_tail_updates(True)
    draft = _validate_and_rewrite_draft(state, c, draft, msgs, max_tok)
    if revision_history:
        state["rejected_draft_history"] = revision_history + [user_msg]
    state["current_draft"] = draft
    wcb.append({"role":"assistant","content":f"*事件草稿：*\n\n{draft}"})
    return state, wcb, "", gr.update(), gr.update(), *_writing_tail_updates(True)

def handle_accept_draft(state, wcb):
    if state.get("pending_actor") in SETTING_UPDATE_ACTORS:
        return _apply_setting_update(state, wcb)
    if not state.get("current_draft"): return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
    commit_info = _commit_current_draft(state, wcb)
    if commit_info and commit_info.get("had_revision"):
        title = "场景完成大纲已保存" if commit_info.get("is_skip") else "事件已保存"
        _replace_revision_ui_with_accepted(wcb, commit_info, title=title)
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
    if commit_info and commit_info.get("is_skip"):
        wcb.append({"role":"assistant","content":"*场景完成大纲已保存。建议点击「结束本幕」进行场景结算。*"})
    else:
        wcb.append({"role":"assistant","content":"*事件已保存。导演可输入评价或 /k 继续推演下一事件。*"})
    return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)

def handle_regenerate_draft(state, wcb):
    ui = state.get("pending_turn_input","") or state.get("pending_user_input","")
    if not ui: return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
    actor = state.get("pending_actor","")
    if actor in SETTING_UPDATE_ACTORS:
        return _regenerate_setting_update(state, wcb)
    c = get_client(state)
    _refresh_scene_support_context(state, c)
    if actor == "__god__":
        purpose = "god"
    elif actor == "__skip__":
        purpose = "synopsis"
    else:
        purpose = "event"
    revision_history = list(state.get("rejected_draft_history", []))
    msgs = _build_story_api_messages(state, ui, purpose=purpose, revision_history=revision_history)
    if wcb and wcb[-1]["role"]=="assistant": wcb.pop()
    max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS), is_director=True)
    draft, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok)
    if e:
        wcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)
    if actor != "__skip__":
        draft = _validate_and_rewrite_draft(state, c, draft, msgs, max_tok)
    state["current_draft"] = draft
    wcb.append({"role":"assistant","content":f"*已重新生成：*\n\n{draft}"})
    return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)

def handle_discard_draft(state, wcb):
    if not state.get("current_draft"):
        wcb.append({"role":"assistant","content":"*当前没有可废弃的草稿。*"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
    _remember_rejected_current_draft(state, wcb, remove_from_ui=True)
    wcb.append({"role":"assistant","content":"*已移除当前回复，请输入修改意见。*"})
    return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)


def handle_delete_last_reply(state, wcb):
    stack = state.get("undo_stack", [])
    if not stack:
        wcb.append({"role":"assistant","content":"*没有可删除的已通过回复。*"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)
    snapshot = stack.pop()
    scene_path = snapshot.get("scene_path", "")
    if scene_path:
        write_file_raw(scene_path, snapshot.get("scene_before", ""))
    state["writing_history"] = [dict(m) for m in snapshot.get("writing_history_before", [])]
    state["scene_summary"] = snapshot.get("scene_summary_before", "")
    state["scene_summary_history_count"] = snapshot.get("scene_summary_history_count_before", 0)
    state["fact_pack"] = snapshot.get("fact_pack_before", {})
    state["fact_pack_history_count"] = snapshot.get("fact_pack_history_count_before", -1)
    state["fact_pack_scene"] = snapshot.get("fact_pack_scene_before", "")
    state["accepted_event_total"] = snapshot.get("accepted_event_total_before", _accepted_event_count(state))
    _clear_pending_draft(state)
    _clear_revision_state(state)
    ui_start = snapshot.get("ui_start")
    if isinstance(ui_start, int) and 0 <= ui_start <= len(wcb):
        del wcb[ui_start:]
    wcb.append({"role":"assistant","content":"*已删除上一个通过回复，并回滚正式剧情记录。*"})
    return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)


def handle_pass_turn(state, wcb):
    skipped = turn_label(state)
    advance_turn(state)
    _clear_pending_draft(state)
    _clear_revision_state(state)
    wcb.append({"role":"assistant","content":f"*{skipped} 跳过本回合。下一回合：{turn_label(state)}。*"})
    return state, wcb, "", "已跳过当前回合。", writing_mode_label(state), gr.update(), *_writing_tail_updates(False)


def _reset_to_next_scene(state):
    sdir = current_story_dir(state)
    _, snn, snum = get_current_scene_file(sdir)
    state.update({"writing_history":[],"active_characters":[],"scene_premise":"",
        "turn_order":[],"turn_index":0,
        "current_draft":"","pending_user_input":"","scene_num":snum,"scene_name":snn,
        "pending_turn_input":"","pending_actor":"","scene_outline":"",
        "scene_path":os.path.join(sdir,snn),"mode":"PLANNING",
        "planning_history":[],"planning_turns":0})
    _clear_revision_state(state)
    _clear_undo_stack(state)
    _reset_scene_support_context(state)
    return snn


def handle_end_scene(state, wcb):
    sn = state.get("scene_name","")
    c = get_client(state)
    # Save pending draft first
    if state.get("current_draft"):
        commit_info = _commit_current_draft(state, wcb)
        _replace_revision_ui_with_accepted(wcb, commit_info)
    # Generate /s-style outline for remaining scene content
    if state.get("writing_history"):
        _refresh_scene_support_context(state, c, force=True)
        msgs = _build_story_api_messages(
            state,
            "请生成当前场景的完整大纲，从当前剧情点列到本场景收束。",
            purpose="synopsis",
        )
        max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS) * 5, is_director=True)
        outline, e = call_deepseek(c, state["model"], msgs, max_tok=max(max_tok, MAX_TOKENS_SCENE_SYNOPSIS))
        if not e and outline:
            append_file(state.get("scene_path",""), f"**SCENE SYNOPSIS:**\n{outline}")
            wcb.append({"role":"assistant","content":f"*场景大纲已生成并保存。*\n\n{outline}"})
    # Settle
    settlement_msg = _settle_scene(state)
    scene_summary = state.get("_last_scene_summary") or _fallback_scene_summary_from_history(state)
    if _update_outline_file(state, sn, scene_summary):
        settlement_msg = f"{settlement_msg}\n已更新 outline.txt。"
    snn = _reset_to_next_scene(state)
    wcb.append({"role":"assistant","content":f"*已结束 {sn}。*\n\n{settlement_msg}\n\n即将进入 {snn}。"})
    st = f"当前幕: {snn} | 场景: 待定"
    return (state, wcb, st,
        gr.update(visible=True), gr.update(visible=False), gr.update(), *_writing_tail_updates(False))


def handle_save_suggested_characters(state):
    state["pending_new_characters"] = []
    choices = load_characters_from_char_dir(state)
    return state, gr.update(choices=choices, value=None), "故事模式已取消建议角色保存；请使用 /c 创建新角色。"
# ---- Gradio UI ----
def build_ui():
    # Auto-scroll once after story chat content settles.
    autoscroll_js = """
    <script>
    (function(){
        var observedRoots = new WeakSet();
        var timers = new WeakMap();
        var QUIET_MS = 450;

        function scrollNode(el){
            if (!el) return;
            try {
                el.scrollTop = el.scrollHeight;
            } catch (e) {}
        }

        function scrollStoryRoot(root){
            scrollNode(root);
            root.querySelectorAll('.bubble-wrap, [role="log"], .overflow-y-auto, [style*="overflow-y: auto"], [style*="overflow: auto"]').forEach(scrollNode);
            var parent = root.parentElement;
            for (var i = 0; parent && i < 5; i++) {
                if (parent.scrollHeight > parent.clientHeight) scrollNode(parent);
                parent = parent.parentElement;
            }
        }

        function scheduleOneScroll(root){
            if (timers.has(root)) clearTimeout(timers.get(root));
            timers.set(root, setTimeout(function(){
                requestAnimationFrame(function(){ scrollStoryRoot(root); });
                timers.delete(root);
            }, QUIET_MS));
        }

        function observeRoot(root){
            if (!root || observedRoots.has(root)) return;
            observedRoots.add(root);
            var observer = new MutationObserver(function(mutations){
                var changed = mutations.some(function(m){
                    return m.type === "childList" || m.type === "characterData";
                });
                if (changed) scheduleOneScroll(root);
            });
            observer.observe(root, {childList:true, subtree:true, characterData:true});
        }

        function observeStoryChats(){
            document.querySelectorAll('.story-chatbot').forEach(observeRoot);
        }

        var discoveryObserver = new MutationObserver(observeStoryChats);

        function start(){
            observeStoryChats();
            discoveryObserver.observe(document.body, {childList:true, subtree:true});
        }

        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", start);
        } else {
            start();
        }
    })();
    </script>
    """
    css = """
    .draft-box { font-size:20px!important; line-height:1.9!important; }
    .status-bar { font-weight:bold; font-size:15px!important; }
    .chatbot, .chatbot p, .chatbot span, .chatbot li {
        font-size:17px!important;
        line-height:1.75!important;
    }
    .story-chatbot, .story-chatbot p, .story-chatbot span, .story-chatbot li,
    .story-chatbot .message, .story-chatbot .prose {
        font-size:19px!important;
        line-height:1.85!important;
    }
    button {
        padding:2px 8px!important;
        font-size:12px!important;
        min-height:24px!important;
        line-height:1.15!important;
    }
    textarea, input[type="text"] { font-size:15px!important; }
    label { font-size:14px!important; }
    """
    with gr.Blocks(title="DeepSeek 互动小说工作台", css=css, theme=gr.themes.Soft(), head=autoscroll_js) as app:
        state = gr.State(make_state())
        gr.Markdown("# DeepSeek 互动小说工作台\n### 构建世界与角色，然后按回合推进故事")

        with gr.Tabs():
            # Tab 1: Setup
            with gr.Tab("启动 / API"):
                gr.Markdown("### 启动设置")
                with gr.Row():
                    dir_dd = gr.Dropdown(label="故事目录（Stories/剧本文件夹）", choices=list_dirs(), value=DEFAULT_STORY_DIR)
                    model_dd = gr.Dropdown(label="模型", choices=MODEL_CHOICES, value=DEFAULT_MODEL)
                with gr.Row():
                    thinking_dd = gr.Dropdown(label="思考模式", choices=THINKING_TYPE_CHOICES, value=DEFAULT_THINKING_TYPE)
                    effort_dd = gr.Dropdown(label="思考强度", choices=REASONING_EFFORT_CHOICES, value=DEFAULT_REASONING_EFFORT)
                api_box = gr.Textbox(label="DeepSeek API Key", value="", type="password", placeholder="请输入 DeepSeek API Key（只保存在内存中，不写入磁盘）")
                with gr.Row():
                    init_btn = gr.Button("开始", variant="primary")
                    init_status = gr.Textbox(label="状态", value="点击初始化", interactive=False)
                wstat = gr.Textbox(label="项目状态", interactive=False)
                dir_dd.change(fn=handle_change_dir, inputs=[state,dir_dd], outputs=[state])
                api_box.change(fn=handle_change_key, inputs=[state,api_box], outputs=[state])
                model_dd.change(fn=handle_change_model, inputs=[state,model_dd], outputs=[state])
                thinking_dd.change(fn=handle_change_thinking_type, inputs=[state,thinking_dd], outputs=[state])
                effort_dd.change(fn=handle_change_reasoning_effort, inputs=[state,effort_dd], outputs=[state])

            # Tab 2: World
            with gr.Tab("构建模式 - 世界"):
                gr.Markdown("### 世界构建\n输入提示词生成世界观，或输入修改意见让 AI 修改。修改满意后点击保存。")
                wconcept = gr.Textbox(label="世界提示词 / 修改意见", placeholder="生成：描述你想要的世界...\n修改：输入修改意见，AI 修改当前世界观...", lines=3)
                with gr.Row():
                    genw_btn = gr.Button("生成/修改世界观", variant="primary")
                    savew_btn = gr.Button("接受/保存世界观")
                wdisplay = gr.Textbox(label="世界观内容（可编辑）", lines=15, interactive=True)
                wmsg = gr.Textbox(label="状态", interactive=False)
                genw_btn.click(fn=handle_world_generate_or_revise, inputs=[state,wconcept,wdisplay], outputs=[state,wdisplay,wmsg])
                savew_btn.click(fn=handle_save_world, inputs=[state,wdisplay], outputs=[state,wdisplay,wmsg])

            # Tab 3: Characters
            with gr.Tab("构建模式 - 角色"):
                gr.Markdown("### 角色构建\n**未选角色**时输入提示词生成新角色；**先选择一个角色**后再输入，则用 AI 修改当前角色。AI 读取全部已有角色避免矛盾。文件名默认 char1, char2...，可手动修改。")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**已有角色**")
                        clist = gr.Dropdown(label="角色", choices=[], interactive=True)
                        with gr.Row():
                            ref_btn = gr.Button("刷新", size="sm")
                            del_btn = gr.Button("删除选中角色", variant="stop", size="sm")
                        cmsg = gr.Textbox(label="状态", interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown("**创建 / 修改**")
                        cdesc = gr.Textbox(label="角色提示词 / 修改意见", placeholder="未选角色时：描述角色身份、性格、能力...  \n已选角色时：输入修改意见，AI 会基于全部已有角色进行修改...", lines=3)
                        genc_btn = gr.Button("生成角色", variant="primary")
                        char_name = gr.Textbox(label="角色文件名", placeholder="char1", value="", interactive=True)
                        with gr.Row():
                            savec_btn = gr.Button("保存角色")
                            rename_btn = gr.Button("重命名")
                            batch_btn = gr.Button("批量保存")
                        cdisplay = gr.Textbox(label="角色档案（可编辑）", lines=12, interactive=True)
                genc_btn.click(fn=handle_generate_character, inputs=[state,cdesc,cdisplay,char_name], outputs=[state,cdisplay,char_name,cmsg,clist])
                clist.change(fn=handle_load_character, inputs=[state,clist,char_name], outputs=[state,cdisplay,char_name,cmsg])
                savec_btn.click(fn=handle_save_character, inputs=[state,char_name,cdisplay], outputs=[state,char_name,cdisplay,cmsg,clist])
                rename_btn.click(fn=handle_rename_character, inputs=[state,char_name], outputs=[state,char_name,cmsg,clist])
                batch_btn.click(fn=handle_batch_save_characters, inputs=[state,cdisplay], outputs=[state,cdisplay,cmsg,clist])
                ref_btn.click(fn=handle_refresh_chars, inputs=[state], outputs=[state,clist])
                del_btn.click(fn=handle_delete_character, inputs=[state,clist], outputs=[state,clist,cdisplay,cmsg])
                init_btn.click(fn=handle_init, inputs=[state], outputs=[state,wstat,init_status,clist,wdisplay,wmsg])

            # Tab 4: Write
            with gr.Tab("故事模式"):
                gr.Markdown("### 故事推演")
                gr.Markdown(command_help_text())
                mode_text = gr.Textbox(label="模式", value="导演会议 (0/20)", interactive=False, elem_classes=["status-bar"])
                status_bar = gr.Textbox(label="当前幕信息", value="当前幕: scene_01.txt | 场景: 待定", interactive=False)

                # Reply chars slider — always visible, controls token budget for all AI replies
                reply_chars = gr.Slider(
                    label="回复字数上限（中文字符）",
                    minimum=50, maximum=5000, value=DEFAULT_REPLY_CHARS, step=50,
                )

                # Planning Panel (Simplified)
                initial_scenes = list_scene_files(DEFAULT_STORY_DIR)
                plan_panel = gr.Column(visible=True)
                with plan_panel:
                    gr.Markdown("#### 场景设定")
                    with gr.Row():
                        scene_select = gr.Dropdown(
                            label="继续已有场景",
                            choices=initial_scenes,
                            value=initial_scenes[-1] if initial_scenes else None,
                            interactive=True,
                            scale=2,
                        )
                        refresh_scenes_btn = gr.Button("刷新场景", scale=1)
                        continue_scene_btn = gr.Button("继续选中场景 →", variant="secondary", scale=1)
                    continue_scene_msg = gr.Textbox(label="继续状态", value="", interactive=False)
                    scene_brief = gr.Textbox(
                        label="场景简描",
                        placeholder="可留空直接开始。若填写，可简要描述本幕：出场角色、地点、开场状况。",
                        lines=3,
                    )
                    with gr.Row():
                        quick_start_btn = gr.Button("快速开始 →", variant="primary", scale=2)

                # Writing Panel
                write_panel = gr.Column(visible=False)
                with write_panel:
                    gr.Markdown("#### 写作模式：AI 推演小事件，导演评价")
                    write_cb = gr.Chatbot(label="剧情", type="messages", height=450, value=[], elem_classes=["story-chatbot"])
                    with gr.Row():
                        write_input = gr.Textbox(label="导演评价 / 指令", placeholder="输入评价；/k 通过；/u 强制剧情；/c更新角色卡；/w更新世界设定...", scale=3)
                        write_btn = gr.Button("发送", variant="primary", scale=1)
                        k_btn = gr.Button("/k 通过", variant="secondary", scale=1)
                        s_btn = gr.Button("/s 大纲", variant="secondary", scale=1)
                    act_row = gr.Row(visible=True)
                    with act_row:
                        accept_btn = gr.Button("接受并保存", variant="primary")
                        regen_btn = gr.Button("重新生成")
                        discard_btn = gr.Button("", visible=False)
                        delete_last_btn = gr.Button("删除上一个回复", variant="stop")
                        end_sc_btn = gr.Button("结束本幕（生成大纲并结算）")

                # Wiring: Planning (Simplified)
                quick_start_btn.click(fn=handle_quick_start, inputs=[state,scene_brief],
                    outputs=[state,write_cb,mode_text,status_bar,plan_panel,write_panel])
                refresh_scenes_btn.click(fn=handle_refresh_scene_choices, inputs=[state],
                    outputs=[scene_select,continue_scene_msg])
                continue_scene_btn.click(fn=handle_continue_scene, inputs=[state,scene_select],
                    outputs=[state,write_cb,mode_text,status_bar,plan_panel,write_panel,continue_scene_msg])
                dir_dd.change(fn=handle_scene_dir_choices, inputs=[dir_dd], outputs=[scene_select])
                init_btn.click(fn=handle_refresh_scene_choices, inputs=[state],
                    outputs=[scene_select,continue_scene_msg])

                # Wiring: Writing
                wout = [state,write_cb,write_input,mode_text,accept_btn,regen_btn,discard_btn,reply_chars]
                aout = [state,write_cb,write_input,mode_text,accept_btn,regen_btn,discard_btn,reply_chars]
                sout = [state,write_cb,status_bar,plan_panel,write_panel,accept_btn,regen_btn,discard_btn,reply_chars]

                reply_chars.change(fn=handle_reply_chars_change, inputs=[state,reply_chars], outputs=[state])

                write_btn.click(fn=handle_writing_send, inputs=[state,write_input,write_cb], outputs=wout)
                write_input.submit(fn=handle_writing_send, inputs=[state,write_input,write_cb], outputs=wout)
                k_btn.click(fn=lambda s,cb: handle_writing_send(s, "/k", cb), inputs=[state,write_cb], outputs=wout)
                s_btn.click(fn=lambda s,cb: handle_writing_send(s, "/s", cb), inputs=[state,write_cb], outputs=wout)
                accept_btn.click(fn=handle_accept_draft, inputs=[state,write_cb], outputs=aout)
                regen_btn.click(fn=handle_regenerate_draft, inputs=[state,write_cb], outputs=aout)
                delete_last_btn.click(fn=handle_delete_last_reply, inputs=[state,write_cb], outputs=aout)
                end_sc_btn.click(fn=handle_end_scene, inputs=[state,write_cb], outputs=sout)

        gr.Markdown("---\n*Powered by DeepSeek API | Gradio UI*")
    return app

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=find_free_port(), inbrowser=True, share=False)
