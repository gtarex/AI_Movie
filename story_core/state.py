import os

from config import STORY_DIR as DEFAULT_STORY_DIR, DEFAULT_REPLY_CHARS, MAX_HISTORY_PAIRS
from .api import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_THINKING_TYPE


UNDO_STACK_LIMIT = 10
MAX_FACT_REWRITE_ATTEMPTS = 2
STORY_HISTORY_CONTEXT_CHARS = 20000


class StoryStateService:
    def make_state(self):
        return {
            "story_dir": DEFAULT_STORY_DIR,
            "api_key": "",
            "model": DEFAULT_MODEL,
            "thinking_type": DEFAULT_THINKING_TYPE,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "world_bg": "",
            "characters": {},
            "scene_num": 1,
            "scene_name": "scene_01.txt",
            "scene_path": os.path.join(DEFAULT_STORY_DIR, "scene_01.txt"),
            "scene_premise": "",
            "scene_outline": "",
            "active_characters": [],
            "mode": "PLANNING",
            "turn_order": [],
            "turn_index": 0,
            "planning_history": [],
            "planning_turns": 0,
            "writing_history": [],
            "current_draft": "",
            "pending_user_input": "",
            "pending_turn_input": "",
            "pending_actor": "",
            "pending_new_characters": [],
            "reply_char_count": DEFAULT_REPLY_CHARS,
            "rejected_draft_history": [],
            "revision_ui_start": None,
            "revision_base_actor": "",
            "undo_stack": [],
            "pending_setting_update": None,
            "scene_summary": "",
            "scene_summary_history_count": 0,
            "scene_summary_history_chars": 0,
            "fact_pack": {},
            "fact_pack_history_count": -1,
            "fact_pack_scene": "",
            "accepted_event_total": 0,
        }

    def build_turn_order(self, active_characters):
        return [character for character in active_characters if character]

    def get_current_actor(self, state):
        order = state.get("turn_order", [])
        if not order:
            return None
        return order[state.get("turn_index", 0) % len(order)]

    def advance_turn(self, state):
        order = state.get("turn_order", [])
        state["turn_index"] = (state.get("turn_index", 0) + 1) % len(order) if order else 0
        return state

    def turn_label(self, state):
        return self.get_current_actor(state) or "旁白 / 自由行动"

    def writing_mode_label(self, state):
        return "故事模式 | 导演评价中"

    def command_help_text(self):
        return (
            "### 指令速查\n"
            "- **场景设定**：在「场景简描」中简要描述出场角色、地点和开场状况，点击「快速开始」即可直接进入故事模式；留空点击则无大纲开始，由你第一段输入指定开场。\n"
            "- **写作模式**：AI 自行推演事件，你作为导演评价；可用下方「回复长度预算」滑块控制 API 输出 token 预算。\n"
            "- `/k`：**通过当前事件**，AI 自动保存并继续推演下一事件。\n"
            "- `/s`：**场景大纲模式**，AI 自行生成当前场景的完整大纲（8-14条条目列表）。\n"
            "- `/u 你的修正`：上帝指令，用来修正剧情错误，不消耗回合。\n"
            "- `/c角色设定修改`：临时更新或创建角色卡；确认后写入角色卡并重新载入，不写入剧情记录。\n"
            "- `/w世界设定修改`：临时更新世界设定；确认后写入 World.txt 并重新载入，不写入剧情记录。\n"
            "- **评价/反馈**：输入你的想法，AI 会参考后重新生成当前事件。\n"
            "- `结束本幕`：进行结算（场景摘要、角色记忆/世界观），需要场景大纲时请先使用 `/s`。\n"
        )

    def clear_revision_state(self, state):
        state["rejected_draft_history"] = []
        state["revision_ui_start"] = None
        state["revision_base_actor"] = ""
        return state

    def clear_pending_draft(self, state):
        state["current_draft"] = ""
        state["pending_user_input"] = ""
        state["pending_turn_input"] = ""
        state["pending_actor"] = ""
        state["pending_setting_update"] = None
        return state

    def trim_writing_history(self, state, max_history_pairs):
        return state

    def clear_undo_stack(self, state):
        state["undo_stack"] = []
        return state


state_service = StoryStateService()


def make_state():
    return state_service.make_state()


def build_turn_order(active_characters):
    return state_service.build_turn_order(active_characters)


def get_current_actor(state):
    return state_service.get_current_actor(state)


def advance_turn(state):
    return state_service.advance_turn(state)


def turn_label(state):
    return state_service.turn_label(state)


def writing_mode_label(state):
    return state_service.writing_mode_label(state)


def command_help_text():
    return state_service.command_help_text()


def _clear_revision_state(state):
    return state_service.clear_revision_state(state)


def _clear_pending_draft(state):
    return state_service.clear_pending_draft(state)


def _trim_writing_history(state, max_history_pairs=MAX_HISTORY_PAIRS):
    return state_service.trim_writing_history(state, max_history_pairs)


def _clear_undo_stack(state):
    return state_service.clear_undo_stack(state)
