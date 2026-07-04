import os
import re

import gradio as gr

from .api import (
    normalize_model,
    normalize_reasoning_effort,
    normalize_thinking_type,
    reset_client,
    sync_api_settings,
)
from .characters import extract_character_name
from .scene_records import _parse_scene_record
from .state import _clear_revision_state, _clear_undo_stack, build_turn_order, writing_mode_label
from .storage import (
    current_story_dir,
    ensure_char_dir,
    get_current_scene_file,
    list_scene_files,
    normalize_story_dir,
    outline_path,
    read_file,
    read_file_raw,
    scene_number_from_name,
)
from .story_context import _accepted_event_count, _reset_scene_support_context
from .text_utils import limit_chars


class SetupHandlers:
    def outline_summary_for_scene(self, state, scene_name):
        key = os.path.splitext(scene_name or "")[0]
        for line in read_file_raw(outline_path(current_story_dir(state))).splitlines():
            match = re.match(rf"^{re.escape(key)}\s*[:：]\s*(.+)$", line.strip())
            if match:
                return match.group(1).strip()
        return ""

    def scene_premise_from_loaded_record(self, state, scene_name, parsed):
        outline_summary = self.outline_summary_for_scene(state, scene_name)
        if outline_summary:
            return f"继续已有场景：{outline_summary}"
        first_story = (parsed.get("first_story") or "").replace("\n", " ")
        if first_story:
            return f"继续已有场景：{limit_chars(first_story, 180)}"
        return "继续已有场景"

    def active_characters_for_scene(self, state, raw):
        chars = state.get("characters", {}) or {}
        matched = []
        for codename in sorted(chars.keys()):
            profile = chars.get(codename, "")
            name = extract_character_name(profile)
            probes = [codename, name]
            if name and len(name) >= 3:
                probes.append(name[-2:])
            if any(probe and probe in raw for probe in probes):
                matched.append(codename)
        return matched or sorted(chars.keys())

    def load_characters_from_char_dir(self, state):
        story_dir = current_story_dir(state)
        char_dir = ensure_char_dir(story_dir)
        characters = {}
        for filename in sorted(os.listdir(char_dir)):
            if filename.endswith(".txt") and not filename.startswith("."):
                codename = os.path.splitext(filename)[0]
                characters[codename] = read_file(os.path.join(char_dir, filename))
        state["characters"] = characters
        return list(characters.keys())

    def handle_init(self, state):
        story_dir = current_story_dir(state)
        os.makedirs(story_dir, exist_ok=True)
        world_bg = read_file(os.path.join(story_dir, "World.txt"))
        choices = self.load_characters_from_char_dir(state)
        chars = state.get("characters", {})
        scene_path, scene_name, scene_num = get_current_scene_file(story_dir)
        for key in ["world_bg", "characters", "scene_num", "scene_name", "scene_path", "scene_premise", "active_characters"]:
            state[key] = (
                world_bg if key == "world_bg" else
                chars if key == "characters" else
                scene_num if key == "scene_num" else
                scene_name if key == "scene_name" else
                scene_path if key == "scene_path" else
                "" if key == "scene_premise" else
                []
            )
        state["turn_order"] = []
        state["turn_index"] = 0
        state.update({
            "mode": "PLANNING",
            "planning_history": [],
            "planning_turns": 0,
            "writing_history": [],
            "current_draft": "",
            "pending_user_input": "",
            "pending_turn_input": "",
            "pending_actor": "",
            "pending_new_characters": [],
            "scene_outline": "",
        })
        _clear_revision_state(state)
        _clear_undo_stack(state)
        _reset_scene_support_context(state)
        world_status = f"世界观: {'已载入' if world_bg else '未找到'} | 角色数: {len(chars)} | 下一幕: {scene_name}"
        world_text = world_bg if world_bg else ""
        world_msg = "已载入现有世界观" if world_bg else "尚未创建世界观，可输入提示词后点击生成"
        return state, world_status, f"初始化完成：{world_status}", gr.update(choices=list(chars.keys()), value=None), world_text, world_msg

    def handle_refresh_chars(self, state):
        choices = self.load_characters_from_char_dir(state)
        return state, gr.update(choices=choices, value=None)

    def handle_refresh_scene_choices(self, state):
        scenes = list_scene_files(current_story_dir(state))
        value = scenes[-1] if scenes else None
        msg = f"已找到 {len(scenes)} 个已有场景。" if scenes else "当前故事目录没有可继续的场景。"
        return gr.update(choices=scenes, value=value), msg

    def handle_scene_dir_choices(self, directory_name):
        scenes = list_scene_files(normalize_story_dir(directory_name))
        return gr.update(choices=scenes, value=scenes[-1] if scenes else None)

    def handle_continue_scene(self, state, selected_scene):
        story_dir = current_story_dir(state)
        scenes = list_scene_files(story_dir)
        if not selected_scene or selected_scene not in scenes:
            wcb = [{"role": "assistant", "content": "请选择一个已有场景文件。"}]
            return state, wcb, "导演会议", f"当前幕: {state.get('scene_name','scene_01.txt')} | 场景: 待定", gr.update(visible=True), gr.update(visible=False), "请选择一个已有场景文件。"

        world_bg = read_file(os.path.join(story_dir, "World.txt"))
        state["world_bg"] = world_bg
        self.load_characters_from_char_dir(state)

        scene_path = os.path.join(story_dir, selected_scene)
        parsed = _parse_scene_record(scene_path, selected_scene)
        premise = self.scene_premise_from_loaded_record(state, selected_scene, parsed)
        active = self.active_characters_for_scene(state, parsed.get("raw", ""))

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
            state["scene_summary_history_chars"] = sum(len(str(msg.get("content", ""))) for msg in state.get("writing_history", []))

        record_count = len(parsed.get("writing_history", []))
        chars_str = ", ".join(active) if active else "未指定"
        wcb = parsed.get("ui_messages", [])
        wcb.append({"role": "assistant", "content": f"*已跳过导演会议，继续 {selected_scene}。已恢复 {record_count} 条正式记录。登场角色: {chars_str}。*"})
        if parsed.get("summaries"):
            wcb.append({"role": "assistant", "content": "*提示：该场景文件已包含结算摘要，如需继续请确认这是你想续写的版本。*"})
        status = f"当前幕: {selected_scene} | 场景: {premise[:60]}..."
        return state, wcb, writing_mode_label(state), status, gr.update(visible=False), gr.update(visible=True), f"已继续 {selected_scene}。"

    def handle_change_dir(self, state, directory_name):
        state["story_dir"] = normalize_story_dir(directory_name)
        return state

    def handle_change_key(self, state, api_key):
        state["api_key"] = api_key
        reset_client()
        return state

    def handle_change_model(self, state, model):
        state["model"] = normalize_model(model)
        sync_api_settings(state)
        return state

    def handle_change_thinking_type(self, state, thinking_type):
        state["thinking_type"] = normalize_thinking_type(thinking_type)
        sync_api_settings(state)
        return state

    def handle_change_reasoning_effort(self, state, effort):
        state["reasoning_effort"] = normalize_reasoning_effort(effort)
        sync_api_settings(state)
        return state


setup_handlers = SetupHandlers()


def _outline_summary_for_scene(state, scene_name):
    return setup_handlers.outline_summary_for_scene(state, scene_name)


def _scene_premise_from_loaded_record(state, scene_name, parsed):
    return setup_handlers.scene_premise_from_loaded_record(state, scene_name, parsed)


def _active_characters_for_scene(state, raw):
    return setup_handlers.active_characters_for_scene(state, raw)


def load_characters_from_char_dir(state):
    return setup_handlers.load_characters_from_char_dir(state)


def handle_init(state):
    return setup_handlers.handle_init(state)


def handle_refresh_chars(state):
    return setup_handlers.handle_refresh_chars(state)


def handle_refresh_scene_choices(state):
    return setup_handlers.handle_refresh_scene_choices(state)


def handle_scene_dir_choices(directory_name):
    return setup_handlers.handle_scene_dir_choices(directory_name)


def handle_continue_scene(state, selected_scene):
    return setup_handlers.handle_continue_scene(state, selected_scene)


def handle_change_dir(state, directory_name):
    return setup_handlers.handle_change_dir(state, directory_name)


def handle_change_key(state, api_key):
    return setup_handlers.handle_change_key(state, api_key)


def handle_change_model(state, model):
    return setup_handlers.handle_change_model(state, model)


def handle_change_thinking_type(state, thinking_type):
    return setup_handlers.handle_change_thinking_type(state, thinking_type)


def handle_change_reasoning_effort(state, effort):
    return setup_handlers.handle_change_reasoning_effort(state, effort)
