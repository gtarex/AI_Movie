import json
import os
import re

import gradio as gr

from config import MAX_PLANNING_TURNS, MAX_TOKENS_SCENE_SETUP
from .api import API_TIMEOUT_SUPPORT, call_deepseek, get_client
from .setup_handlers import load_characters_from_char_dir
from .state import _clear_revision_state, _clear_undo_stack, build_turn_order, writing_mode_label
from .storage import current_story_dir, read_file
from .story_context import _build_character_context, _outline_context_block, _reset_scene_support_context


class PlanningHandlers:
    def handle_planning_send(self, state, user_msg, planning_chatbot):
        if not user_msg or not user_msg.strip():
            return state, planning_chatbot, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        world_bg = state.get("world_bg", "")
        scene_name = state.get("scene_name", "scene_01.txt")
        command = user_msg.strip()
        is_k = command.lower() == "/k"
        is_god = command.lower().startswith("/u")

        if is_k:
            known_chars = state.get("characters", {})
            existing_chars = list(known_chars.keys())
            char_profiles = _build_character_context(state, active_only=False)
            outline_block = _outline_context_block(state)
            system_prompt = (
                "你是一个互动小说的 AI 写手。用户使用 /k 命令，要求你自行决定下一幕的场景并提供开头剧情。"
                f"\n世界观：\n{world_bg}{outline_block}\n\n当前幕：{scene_name}\n"
                f"\n[已有角色档案]{char_profiles}\n"
                "\n请输出一个 JSON 对象（不要 markdown），包含："
                "\n- premise: 中文描述本幕的地点、时间、开场矛盾和故事走向"
                "\n- characters: 从已有角色中选出的登场角色 codename 列表"
                "\n- opening_hint: 一句简短的开场提示（用于写作模式的起始事件）"
                "\n如果无需指定角色或无可选角色，characters 可以为空列表。"
            )
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "/k 请自行决定本幕场景并开始写作。"}]
            planning_chatbot.append({"role": "user", "content": "/k（导演要求 AI 自由决定场景）"})
            client = get_client(state)
            raw, err = call_deepseek(client, state["model"], msgs)
            if err:
                planning_chatbot.append({"role": "assistant", "content": f"[错误] {err}"})
                return state, planning_chatbot, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            premise = "待定场景"
            chars = []
            opening = ""
            try:
                match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    premise = data.get("premise", "待定场景")
                    chars = data.get("characters", []) or []
                    opening = data.get("opening_hint", "")
            except Exception:
                pass
            state["planning_history"].append({"role": "user", "content": "/k"})
            state["planning_history"].append({"role": "assistant", "content": raw})
            planning_chatbot.append({"role": "assistant", "content": raw})
            valid_chars = [codename for codename in chars if codename in known_chars] or chars
            self._enter_writing_mode(state, premise, valid_chars)
            planning_chatbot.append({"role": "assistant", "content": f"**AI 自行决定场景：**\n场景: {premise}\n角色: {', '.join(valid_chars) if valid_chars else '未指定'}\n\n--- 进入故事模式 ---"})
            writing_chatbot = [{"role": "assistant", "content": f"**{state['scene_name']}**\n场景: {premise}\n\n请输入 /k 或你的评价来启动第一个事件。{('开头提示：' + opening) if opening else ''}"}]
            status = f"当前幕: {state['scene_name']} | 场景: {premise[:60]}..."
            return state, planning_chatbot, "", writing_chatbot, writing_mode_label(state), status, gr.update(visible=False), gr.update(visible=True)

        char_context = _build_character_context(state, active_only=False)
        outline_block = _outline_context_block(state)
        system_prompt = (
            "你是用户的小说导演会议伙伴。请用中文和用户讨论下一幕的地点、时间、登场人物和开场矛盾。"
            f"\n世界观：\n{world_bg}{outline_block}\n\n当前幕：{scene_name}\n"
            f"{char_context}"
            f"\n最多讨论 {MAX_PLANNING_TURNS} 轮。"
            "\n如果用户输入 /u，这是最高优先级的上帝指令，必须立刻按它修正当前计划。"
            "\n如果用户输入 /k，表示导演让你自行决定场景并直接开始写作。"
        )
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(state.get("planning_history", []))
        msgs.append({"role": "user", "content": f"[上帝指令]\n{command[2:].strip()}" if is_god else command})
        planning_chatbot.append({"role": "user", "content": command})
        client = get_client(state)
        raw, err = call_deepseek(client, state["model"], msgs)
        if err:
            planning_chatbot.append({"role": "assistant", "content": f"[错误] {err}"})
            return state, planning_chatbot, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        state["planning_history"].append({"role": "user", "content": f"[上帝指令]\n{command[2:].strip()}" if is_god else command})
        state["planning_history"].append({"role": "assistant", "content": raw})
        state["planning_turns"] = state.get("planning_turns", 0) + 1
        planning_chatbot.append({"role": "assistant", "content": raw})
        mode_text = f"导演会议 ({state['planning_turns']}/{MAX_PLANNING_TURNS})"
        if state["planning_turns"] >= MAX_PLANNING_TURNS:
            state, planning_chatbot, writing_chatbot, mode_text, status, plan_update, write_update = self.magic_transition(state, planning_chatbot)
            return state, planning_chatbot, "", writing_chatbot, mode_text, status, plan_update, write_update
        return state, planning_chatbot, "", gr.update(), mode_text, gr.update(), gr.update(), gr.update()

    def magic_transition(self, state, planning_chatbot):
        if not state.get("planning_history"):
            planning_chatbot.append({"role": "assistant", "content": "还没有导演会议记录。"})
            return state, planning_chatbot, [], "导演会议", "请先讨论这一幕。", gr.update(visible=True), gr.update(visible=False)
        client = get_client(state)
        char_context = _build_character_context(state, active_only=False)
        system_prompt = (
            "分析这段导演会议。只输出一个合法 JSON 对象，不要 markdown。必须包含 keys: "
            "'premise'（中文字符串，概括地点、时间、目标/冲突）和 'characters'（角色 codename 列表）。"
            f"\n\n[可用角色档案]{char_context}\n"
            'Example: {"premise":"Late night at the bar.","characters":["elara","shadow_knight"]}'
        )
        msgs = [{"role": "system", "content": system_prompt}] + state["planning_history"]
        raw, err = call_deepseek(client, state["model"], msgs, max_tok=300, timeout=API_TIMEOUT_SUPPORT, thinking_type="disabled")
        premise = "待定场景"
        chars = []
        if not err and raw:
            match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    premise = data.get("premise", "待定场景")
                    chars = data.get("characters", [])
                except Exception:
                    premise = raw[:200]
            else:
                premise = raw[:200]
        known_chars = state.get("characters", {})
        valid_chars = [codename for codename in chars if codename in known_chars] or chars
        self._enter_writing_mode(state, premise, valid_chars)
        planning_chatbot.append({"role": "assistant", "content": f"**已提取场景：**\n场景: {premise}\n角色: {', '.join(valid_chars) if valid_chars else '未指定'}\n\n--- 进入故事模式 ---"})
        writing_chatbot = [{"role": "assistant", "content": f"**{state['scene_name']}**\n场景: {premise}\n\n请输入 /k 来让 AI 开始推演第一个事件，或输入你的评价/指引。"}]
        status = f"当前幕: {state['scene_name']} | 场景: {premise[:60]}..."
        return state, planning_chatbot, writing_chatbot, writing_mode_label(state), status, gr.update(visible=False), gr.update(visible=True)

    def handle_start_writing(self, state, planning_chatbot):
        return self.magic_transition(state, planning_chatbot)

    def handle_reply_chars_change(self, state, value):
        if value is not None:
            state["reply_char_count"] = int(value)
        return state

    def writing_tail_updates(self, show_regen=False):
        return gr.update(visible=show_regen), gr.update(visible=False), gr.update(visible=True)

    def handle_no_outline_start(self, state):
        story_dir = current_story_dir(state)
        world_bg = read_file(os.path.join(story_dir, "World.txt"))
        state["world_bg"] = world_bg
        load_characters_from_char_dir(state)
        scene_name = state.get("scene_name", "scene_01.txt")
        active_chars = list(state.get("characters", {}).keys())
        self._enter_writing_mode(state, "", active_chars)
        state["planning_history"] = []
        state["planning_turns"] = 0
        chars_str = ", ".join(active_chars) if active_chars else "未指定"
        writing_chatbot = [{"role": "assistant", "content": f"**{scene_name}**\n无大纲开始。登场角色: {chars_str}。\n\n请直接输入第一段开场/指引，AI 将据此开始推演。"}]
        status = f"当前幕: {scene_name} | 场景: 无大纲开始"
        return state, writing_chatbot, writing_mode_label(state), status, gr.update(visible=False), gr.update(visible=True)

    def handle_quick_start(self, state, scene_brief):
        brief = (scene_brief or "").strip()
        if not brief:
            return self.handle_no_outline_start(state)

        story_dir = current_story_dir(state)
        world_bg = read_file(os.path.join(story_dir, "World.txt"))
        if world_bg:
            state["world_bg"] = world_bg
        load_characters_from_char_dir(state)

        client = get_client(state)
        scene_name = state.get("scene_name", "scene_01.txt")
        char_context = _build_character_context(state, active_only=False)
        outline_block = _outline_context_block(state)
        system_prompt = (
            "你是小说场景构建器。用户给了场景简述，请你内部完成场景设定。"
            "只输出合法JSON，不要markdown。必须包含："
            "premise（中文字符串，补全地点/时间/目标/冲突）、"
            "outline（中文字符串，3-6个关键剧情节点）、"
            "characters（角色codename数组）。"
            f"\n\n[世界观]\n{world_bg}{outline_block}\n\n[已有角色]\n{char_context}\n\n[用户给的场景简述]\n{brief}"
        )
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": brief}]
        raw, err = call_deepseek(client, state["model"], msgs, max_tok=MAX_TOKENS_SCENE_SETUP, timeout=API_TIMEOUT_SUPPORT, thinking_type="disabled")
        premise = brief[:200]
        outline = ""
        chars = []
        if not err and raw:
            try:
                match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    premise = data.get("premise", brief[:200])
                    outline = data.get("outline", "")
                    chars = data.get("characters", []) or []
            except Exception:
                pass

        known_chars = state.get("characters", {})
        valid_chars = [codename for codename in chars if codename in known_chars] or list(known_chars.keys())[:4]
        self._enter_writing_mode(state, premise, valid_chars, scene_outline=outline or premise)
        state["planning_history"] = []
        state["planning_turns"] = 0
        chars_str = ", ".join(valid_chars) if valid_chars else "全部角色"
        writing_chatbot = [{"role": "assistant", "content": f"**{scene_name}**\n场景: {premise}\n\n登场: {chars_str}\n\n请输入评价/指引来启动第一个事件，或输入 /k 让 AI 自行开场。"}]
        status = f"当前幕: {scene_name} | 场景: {premise[:60]}..."
        return state, writing_chatbot, writing_mode_label(state), status, gr.update(visible=False), gr.update(visible=True)

    def handle_skip_planning(self, state, planning_chatbot, manual_premise):
        premise = manual_premise.strip() if manual_premise else "待定场景"
        active_chars = list(state.get("characters", {}).keys())
        self._enter_writing_mode(state, premise, active_chars)
        planning_chatbot.append({"role": "assistant", "content": f"已跳过导演会议。\n场景: {premise}"})
        writing_chatbot = [{"role": "assistant", "content": f"**{state['scene_name']}**\n场景: {premise}\n\n请输入 /k 来让 AI 开始推演第一个事件，或输入你的评价/指引。"}]
        status = f"当前幕: {state['scene_name']} | 场景: {premise[:60]}..."
        return state, planning_chatbot, writing_chatbot, writing_mode_label(state), status, gr.update(visible=False), gr.update(visible=True)

    def _enter_writing_mode(self, state, premise, active_chars, scene_outline=None):
        state["scene_premise"] = premise
        state["scene_outline"] = premise if scene_outline is None else scene_outline
        state["active_characters"] = active_chars
        state["turn_order"] = build_turn_order(active_chars)
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
        return state


planning_handlers = PlanningHandlers()


def handle_planning_send(state, user_msg, planning_chatbot):
    return planning_handlers.handle_planning_send(state, user_msg, planning_chatbot)


def _magic_transition(state, planning_chatbot):
    return planning_handlers.magic_transition(state, planning_chatbot)


def handle_start_writing(state, planning_chatbot):
    return planning_handlers.handle_start_writing(state, planning_chatbot)


def handle_reply_chars_change(state, value):
    return planning_handlers.handle_reply_chars_change(state, value)


def _writing_tail_updates(show_regen=False):
    return planning_handlers.writing_tail_updates(show_regen)


def handle_no_outline_start(state):
    return planning_handlers.handle_no_outline_start(state)


def handle_quick_start(state, scene_brief):
    return planning_handlers.handle_quick_start(state, scene_brief)


def handle_skip_planning(state, planning_chatbot, manual_premise):
    return planning_handlers.handle_skip_planning(state, planning_chatbot, manual_premise)

