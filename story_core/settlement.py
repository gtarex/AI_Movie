import os

import gradio as gr

from config import DEFAULT_REPLY_CHARS, MAX_TOKENS_MEMORY_COMPRESSION
from .api import API_TIMEOUT_DEFAULT, API_TIMEOUT_SUPPORT, call_deepseek, get_client
from .characters import normalize_character_profile
from .drafts import _commit_current_draft, _replace_revision_ui_with_accepted
from .planning_handlers import _writing_tail_updates
from .state import _clear_revision_state, _clear_undo_stack
from .storage import append_file, character_path, current_story_dir, get_current_scene_file, read_file, write_file
from .story_context import (
    _build_character_context,
    _fallback_scene_summary_from_history,
    _reset_scene_support_context,
    _update_outline_file,
)
from .text_utils import extract_json_object, limit_words, story_message_box


class SettlementService:
    def compress_character_if_needed(self, state, codename):
        story_dir = current_story_dir(state)
        path = character_path(story_dir, codename)
        profile = read_file(path)
        if not profile or len(profile) <= 1000:
            return False
        client = get_client(state)
        system_prompt = (
            "你是角色档案压缩器。角色档案包含固定字段和多次追加的 [Memory Update] 记忆块。\n"
            "请保持字段结构不变，但将「过往经历」字段与所有 [Memory Update] 内容融合，"
            "改写成 1-2 段流畅的中文叙述，概括角色迄今最重要的经历和变化。\n"
            "严格控制在 500 中文字以内。其他字段（姓名/性别/年龄/职业/外貌/性格特征/爱好/人际关系/能力）保持不变。\n"
            "只输出完整角色档案，不要解释。"
        )
        raw, err = call_deepseek(client, state["model"], [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请压缩以下角色档案：\n\n{profile}"},
        ], max_tok=MAX_TOKENS_MEMORY_COMPRESSION, timeout=API_TIMEOUT_SUPPORT, thinking_type="disabled")
        if err or not raw:
            return False
        compressed = normalize_character_profile(raw.strip(), codename)
        compressed = limit_words(compressed)
        if compressed:
            write_file(path, compressed)
            state["characters"][codename] = compressed
            return True
        return False

    def settle_scene(self, state):
        history = state.get("writing_history", [])
        state["_last_scene_summary"] = ""
        state["pending_new_characters"] = []
        if not history:
            return "本幕没有已保存剧情，已跳过结算。"

        story_dir = current_story_dir(state)
        history_text = "\n".join([
            f"[{'SYSTEM/PLAYER' if msg['role'] == 'user' else 'AI/STORY'}]: {msg['content']}"
            for msg in history
        ])
        character_contexts = _build_character_context(state, active_only=True)
        if not character_contexts:
            character_contexts = _build_character_context(state, active_only=False)

        existing = sorted(state.get("characters", {}).keys())
        system_prompt = (
            "你是互动小说的连续性编辑。在一幕结束时，你要整理剧情，并判断哪些持久化文件需要更新。\n"
            "只输出合法 JSON 对象，不要 markdown。必须包含这些 key：\n"
            "- scene_summary: 用中文简要总结本幕发生了什么。\n"
            "- world_update: 用中文写重要的新世界观、地点、势力、规则变化；如果没有，写 \"NONE\"。\n"
            "- character_memory_updates: 对象，key 是现有登场角色 codename，value 是 1-3 句中文记忆更新；如果没有，写 \"NONE\"。\n"
            "不要建议或创建新角色；故事模式的新角色只能由导演使用 /c 单独创建。"
        )
        user_prompt = (
            f"[世界观]\n{state.get('world_bg','')}\n\n"
            f"[现有角色 codename]\n{', '.join(existing) if existing else '无'}\n\n"
            f"[登场角色档案]\n{character_contexts if character_contexts else '无'}\n\n"
            f"[当前场景总结]\n{state.get('scene_summary','') or '无'}\n\n"
            f"[本幕记录]\n{history_text}"
        )
        client = get_client(state)
        raw, err = call_deepseek(client, state["model"], [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], max_tok=2500, timeout=API_TIMEOUT_DEFAULT, thinking_type="disabled")
        if err:
            return f"结算失败：{err}"

        settlement = extract_json_object(raw)
        if not settlement:
            return f"结算结果无法解析：{raw[:300]}"

        notes = []
        scene_summary = str(settlement.get("scene_summary", "")).strip()
        if scene_summary:
            state["_last_scene_summary"] = scene_summary
            append_file(state.get("scene_path", ""), f"[Scene Summary]\n{scene_summary}")
            notes.append("已保存本幕摘要。")

        memories = settlement.get("character_memory_updates", {})
        if isinstance(memories, dict):
            for codename, memory in memories.items():
                if codename not in state.get("active_characters", []):
                    continue
                memory = str(memory).strip()
                if not memory or memory.upper() == "NONE":
                    continue
                path = character_path(story_dir, codename)
                if os.path.exists(path):
                    append_file(path, f"[Memory Update]\n{memory}")
                    state["characters"][codename] = read_file(path)
                    notes.append(f"已更新 {codename} 的记忆。")
                    if self.compress_character_if_needed(state, codename):
                        notes.append(f"已压缩 {codename} 的角色档案。")

        world_update = str(settlement.get("world_update", "")).strip()
        if world_update and world_update.upper() != "NONE":
            state["world_bg"] = (state.get("world_bg", "").strip() + f"\n\n[World Update]\n{world_update}").strip()
            write_file(os.path.join(story_dir, "World.txt"), state["world_bg"])
            notes.append("已更新世界观。")

        if not notes:
            notes.append("本幕没有需要写入的角色记忆或世界观更新。")
        return "\n".join(notes)

    def reset_to_next_scene(self, state):
        story_dir = current_story_dir(state)
        _, scene_name, scene_num = get_current_scene_file(story_dir)
        state.update({
            "writing_history": [],
            "active_characters": [],
            "scene_premise": "",
            "turn_order": [],
            "turn_index": 0,
            "current_draft": "",
            "pending_user_input": "",
            "scene_num": scene_num,
            "scene_name": scene_name,
            "pending_turn_input": "",
            "pending_actor": "",
            "scene_outline": "",
            "scene_path": os.path.join(story_dir, scene_name),
            "mode": "PLANNING",
            "planning_history": [],
            "planning_turns": 0,
        })
        _clear_revision_state(state)
        _clear_undo_stack(state)
        _reset_scene_support_context(state)
        return scene_name

    def handle_end_scene(self, state, wcb):
        scene_name = state.get("scene_name", "")
        if state.get("current_draft"):
            commit_info = _commit_current_draft(state, wcb)
            _replace_revision_ui_with_accepted(wcb, commit_info)
            if commit_info and not commit_info.get("had_revision") and wcb:
                for index in range(len(wcb) - 1, -1, -1):
                    if wcb[index].get("role") == "assistant":
                        wcb[index]["content"] = story_message_box(commit_info.get("draft", ""), status="accepted", title="事件已保存")
                        break
        settlement_msg = self.settle_scene(state)
        scene_summary = state.get("_last_scene_summary") or _fallback_scene_summary_from_history(state)
        if _update_outline_file(state, scene_name, scene_summary):
            settlement_msg = f"{settlement_msg}\n已更新 outline.txt。"
        next_scene = self.reset_to_next_scene(state)
        wcb.append({"role": "assistant", "content": f"*已结束 {scene_name}。*\n\n{settlement_msg}\n\n即将进入 {next_scene}。"})
        status = f"当前幕: {next_scene} | 场景: 待定"
        return (
            state,
            wcb,
            status,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(),
            *_writing_tail_updates(False),
        )


settlement_service = SettlementService()


def _compress_character_if_needed(state, codename):
    return settlement_service.compress_character_if_needed(state, codename)


def _settle_scene(state):
    return settlement_service.settle_scene(state)


def _reset_to_next_scene(state):
    return settlement_service.reset_to_next_scene(state)


def handle_end_scene(state, wcb):
    return settlement_service.handle_end_scene(state, wcb)
