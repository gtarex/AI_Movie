#!/usr/bin/env python3
"""DeepSeek Novel Writer - Web UI (Gradio)"""

import os, re, json
import gradio as gr
from config import STORIES_ROOT, STORY_DIR as DEFAULT_STORY_DIR, MAX_HISTORY_PAIRS, MAX_PLANNING_TURNS, MAX_TOKENS_DEFAULT, MAX_TOKENS_WORLD, MAX_TOKENS_CHARACTER_PROFILE, MAX_TOKENS_SCENE_SETUP, MAX_TOKENS_MEMORY_COMPRESSION, MAX_TOKENS_SETTLEMENT, MAX_TOKENS_SCENE_MEMORY, MAX_TOKENS_FACT_PACK, MAX_TOKENS_FACT_CHECK, MAX_TOKENS_PLANNING_PROPOSAL, MAX_TOKENS_PLANNING_CHAT, MAX_TOKENS_STORY_EVENT, MAX_TOKENS_SCENE_SYNOPSIS, DEFAULT_REPLY_CHARS, TOKEN_MULTIPLIER_PER_CHAR, compute_max_tokens
from story_core.api import (
    API_TIMEOUT_DEFAULT,
    API_TIMEOUT_LONG,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING_TYPE,
    MODEL_CHOICES,
    REASONING_EFFORT_CHOICES,
    THINKING_TYPE_CHOICES,
    call_deepseek,
    get_client,
)
from story_core.characters import (
    extract_character_name,
    local_codename_from_text,
    normalize_character_profile,
    sanitize_codename,
    unique_character_codename,
)
from story_core.build_handlers import (
    handle_batch_save_characters,
    handle_delete_character,
    handle_generate_character,
    handle_load_character,
    handle_rename_character,
    handle_save_character,
    handle_save_world,
    handle_world_generate_or_revise,
)
from story_core.drafts import (
    _commit_current_draft,
    _remember_rejected_current_draft,
    _replace_revision_ui_with_accepted,
    _revision_feedback_message,
)
from story_core.planning_handlers import (
    _writing_tail_updates,
    handle_quick_start,
    handle_reply_chars_change,
)
from story_core.settlement import (
    handle_end_scene,
)
from story_core.setup_handlers import (
    handle_change_dir,
    handle_change_key,
    handle_change_model,
    handle_change_reasoning_effort,
    handle_change_thinking_type,
    handle_continue_scene,
    handle_init,
    handle_refresh_chars,
    handle_refresh_scene_choices,
    handle_scene_dir_choices,
    load_characters_from_char_dir,
)
from story_core.state import (
    _clear_pending_draft,
    _clear_revision_state,
    advance_turn,
    command_help_text,
    make_state,
    turn_label,
    writing_mode_label,
)
from story_core.story_context import (
    _accepted_event_count,
    _build_character_context,
    _build_event_prompt,
    _build_story_api_messages,
    _check_context_size,
    _refresh_fact_pack,
    _refresh_scene_support_context,
)
from story_core.storage import (
    character_path,
    current_story_dir,
    find_free_port,
    list_dirs,
    list_scene_files,
    read_file,
    write_file,
    write_file_raw,
)
from story_core.text_utils import extract_json_object, story_message_box

# Monkey-patch Gradio 4.44.1 bug: additionalProperties:true crashes schema parser
import gradio_client.utils as grc_utils
_orig_get_type = grc_utils.get_type
def _patched_get_type(schema):
    if schema is True or schema is False or not isinstance(schema, dict):
        return "string"
    return _orig_get_type(schema)
grc_utils.get_type = _patched_get_type

SETTING_UPDATE_ACTORS = {"__char_update__", "__world_update__"}

def _setting_update_label(kind):
    return "角色卡" if kind == "character" else "世界设定"

def _command_text(command, prefix):
    return (command or "")[len(prefix):].strip()

def _mark_last_draft_accepted(wcb, draft, title="已通过"):
    if not wcb or not draft:
        return wcb
    for index in range(len(wcb) - 1, -1, -1):
        if wcb[index].get("role") == "assistant":
            wcb[index]["content"] = story_message_box(draft, status="accepted", title=title)
            break
    return wcb

def _capture_story_draft_state(state):
    return {
        "current_draft": state.get("current_draft", ""),
        "pending_user_input": state.get("pending_user_input", ""),
        "pending_turn_input": state.get("pending_turn_input", ""),
        "pending_actor": state.get("pending_actor", ""),
        "rejected_draft_history": [dict(m) for m in state.get("rejected_draft_history", [])],
        "revision_ui_start": state.get("revision_ui_start"),
        "revision_base_actor": state.get("revision_base_actor", ""),
    }

def _restore_story_draft_state(state, snapshot):
    snapshot = snapshot or {}
    state["current_draft"] = snapshot.get("current_draft", "")
    state["pending_user_input"] = snapshot.get("pending_user_input", "")
    state["pending_turn_input"] = snapshot.get("pending_turn_input", "")
    state["pending_actor"] = snapshot.get("pending_actor", "")
    state["rejected_draft_history"] = [dict(m) for m in snapshot.get("rejected_draft_history", [])]
    state["revision_ui_start"] = snapshot.get("revision_ui_start")
    state["revision_base_actor"] = snapshot.get("revision_base_actor", "")
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
    wcb.append({"role":"user","content":story_message_box(command, status="draft", title="导演输入")})
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
    wcb.append({"role":"user","content":story_message_box(command, status="draft", title="导演输入")})
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
        had_pending_draft = bool(state.get("current_draft"))
        base_actor = state.get("pending_actor", "")
        if state.get("current_draft"):
            _remember_rejected_current_draft(state, wcb, remove_from_ui=True)
            base_actor = state.get("revision_base_actor", base_actor)
        state["pending_user_input"] = command
        state["pending_turn_input"] = f"[上帝指令]\n{instruction}"
        state["pending_actor"] = base_actor if had_pending_draft or state.get("rejected_draft_history") else "__god__"
        c = get_client(state)
        _refresh_scene_support_context(state, c)
        revision_history = list(state.get("rejected_draft_history", []))
        user_msg = {"role":"user","content":f"/u {instruction}"}
        msgs = _build_story_api_messages(state, user_msg["content"], purpose="god", revision_history=revision_history)
        wcb.append({"role":"user","content":story_message_box(command, status="draft", title="导演输入")})
        max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS), is_director=True)
        draft, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok, timeout=API_TIMEOUT_DEFAULT)
        if e:
            wcb.append({"role":"assistant","content":f"[错误] {e}"})
            return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)
        if revision_history:
            state["rejected_draft_history"] = revision_history + [user_msg]
        state["current_draft"] = draft
        wcb.append({"role":"assistant","content":story_message_box(draft, status="draft", title="上帝指令修正草稿")})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)

    if state.get("pending_actor") in SETTING_UPDATE_ACTORS:
        if is_k or lower_command.startswith("/s"):
            return _cancel_setting_update(state, wcb)
        return _continue_setting_update(state, command, wcb)

    # Handle /s skip-to-synopsis: AI generates full scene outline
    if lower_command.startswith("/s"):
        # Extract optional director comment after /s
        s_comment = command[2:].strip()
        # If a draft is pending, /s is just another temporary revision request.
        if state.get("current_draft"):
            _remember_rejected_current_draft(state, wcb, remove_from_ui=True)
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
        wcb.append({"role":"user","content":story_message_box(user_input_label, status="draft", title="导演输入")})
        _check_context_size(state, wcb)
        max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS) * 5, is_director=True)
        draft, e = call_deepseek(c, state["model"], msgs, max_tok=max(max_tok, MAX_TOKENS_SCENE_SYNOPSIS), timeout=API_TIMEOUT_LONG)
        if e:
            wcb.append({"role":"assistant","content":f"[错误] {e}"})
            return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)
        if revision_history:
            state["rejected_draft_history"] = revision_history + [user_msg]
        state["current_draft"] = draft
        wcb.append({"role":"assistant","content":story_message_box(draft, status="draft", title="场景完成大纲草稿")})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)

    # If /k and there's a pending draft, save it first (director passes on this event)
    if is_k and state.get("current_draft"):
        commit_info = _commit_current_draft(state, wcb)
        _replace_revision_ui_with_accepted(wcb, commit_info)
        if commit_info and not commit_info.get("had_revision"):
            _mark_last_draft_accepted(wcb, commit_info.get("draft"), title="事件已通过")
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
    state["pending_actor"] = state.get("revision_base_actor", "") if revision_history else ""

    c = get_client(state)
    _refresh_scene_support_context(state, c)
    user_msg = {"role":"user","content":user_prompt}
    msgs = _build_story_api_messages(state, user_msg["content"], purpose="event", revision_history=revision_history)
    wcb.append({"role":"user","content":story_message_box(user_input_label, status="draft", title="导演输入")})
    # Context size check
    _check_context_size(state, wcb)
    max_tok = compute_max_tokens(state.get("reply_char_count", DEFAULT_REPLY_CHARS), is_director=True)
    draft, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok, timeout=API_TIMEOUT_DEFAULT)
    if e:
        wcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, wcb, "", gr.update(), gr.update(), *_writing_tail_updates(True)
    if revision_history:
        state["rejected_draft_history"] = revision_history + [user_msg]
    state["current_draft"] = draft
    wcb.append({"role":"assistant","content":story_message_box(draft, status="draft", title="事件草稿")})
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
    if commit_info:
        title = "场景完成大纲已保存" if commit_info.get("is_skip") else "事件已保存"
        _mark_last_draft_accepted(wcb, commit_info.get("draft"), title=title)
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
    draft, e = call_deepseek(c, state["model"], msgs, max_tok=max_tok, timeout=API_TIMEOUT_LONG if purpose == "synopsis" else API_TIMEOUT_DEFAULT)
    if e:
        wcb.append({"role":"assistant","content":f"[错误] {e}"})
        return state, wcb, "", writing_mode_label(state), gr.update(), *_writing_tail_updates(True)
    state["current_draft"] = draft
    wcb.append({"role":"assistant","content":story_message_box(draft, status="draft", title="重新生成草稿")})
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
    state["scene_summary_history_chars"] = snapshot.get("scene_summary_history_chars_before", 0)
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
    .story-ai-draft, .story-ai-accepted { border-radius:8px; padding:10px 12px; margin:2px 0; border:1px solid transparent; color:#1f2933; }
    .story-ai-draft { background:#fff7d6; border-color:#e6c766; }
    .story-ai-accepted { background:#e8f6ed; border-color:#82bd92; }
    .story-ai-status { font-size:13px!important; line-height:1.35!important; font-weight:700; margin-bottom:6px; color:#374151; }
    .story-ai-body { white-space:pre-wrap; }
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

                # Reply length slider — always visible, controls API token budget for all AI replies
                reply_chars = gr.Slider(
                    label="回复长度预算（用于 max_tokens）",
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
