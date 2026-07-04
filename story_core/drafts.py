import json

from .state import _clear_pending_draft, _clear_revision_state, _trim_writing_history, UNDO_STACK_LIMIT
from .storage import append_file, read_file_raw
from .story_context import _accepted_event_count
from .text_utils import story_message_box


class DraftService:
    def push_undo_snapshot(self, state, wcb=None):
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
            "scene_summary_history_chars_before": state.get("scene_summary_history_chars", 0),
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

    def remember_rejected_current_draft(self, state, wcb, remove_from_ui=True):
        draft = (state.get("current_draft") or "").strip()
        if not draft:
            return False
        first_rejection = state.get("revision_ui_start") is None
        revision_history = state.setdefault("rejected_draft_history", [])
        if first_rejection:
            state["revision_base_actor"] = state.get("pending_actor", "")
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
                    "content": original_prompt,
                })
        revision_history.append({
            "role": "assistant",
            "content": draft,
        })
        if remove_from_ui and wcb and wcb[-1].get("role") == "assistant":
            wcb.pop()
        _clear_pending_draft(state)
        return True

    def revision_feedback_message(self, command):
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
                "请基于上一版和修改意见重写当前草稿。"
            ),
        }

    def commit_current_draft(self, state, wcb=None):
        if not state.get("current_draft"):
            return None
        self.push_undo_snapshot(state, wcb)
        ui = state.get("pending_user_input", "")
        turn_input = state.get("pending_turn_input", "") or ui
        actor = state.get("pending_actor", "")
        draft = state.get("current_draft", "")
        is_god = actor == "__god__"
        is_skip = actor == "__skip__"
        story_label = "CORRECTION" if is_god else ("SCENE SYNOPSIS" if is_skip else "EVENT")
        had_revision = bool(state.get("rejected_draft_history"))
        revision_ui_start = state.get("revision_ui_start")
        append_file(state.get("scene_path", ""), f"**{story_label}:**\n{draft}")
        if had_revision:
            state["writing_history"].append({"role": "assistant", "content": draft})
        else:
            state["writing_history"].append({"role": "user", "content": turn_input})
            state["writing_history"].append({"role": "assistant", "content": draft})
        state["accepted_event_total"] = int(state.get("accepted_event_total") or max(_accepted_event_count(state) - 1, 0)) + 1
        _trim_writing_history(state)
        _clear_pending_draft(state)
        _clear_revision_state(state)
        return {"draft": draft, "is_skip": is_skip, "had_revision": had_revision, "revision_ui_start": revision_ui_start}

    def replace_revision_ui_with_accepted(self, wcb, commit_info, title="事件已保存"):
        if not commit_info or not commit_info.get("had_revision"):
            return wcb
        start = commit_info.get("revision_ui_start")
        if isinstance(start, int) and 0 <= start <= len(wcb):
            del wcb[start:]
        wcb.append({"role": "assistant", "content": story_message_box(commit_info.get("draft", ""), status="accepted", title=f"{title}（已清理废弃版本）")})
        return wcb


drafts = DraftService()


def _push_undo_snapshot(state, wcb=None):
    return drafts.push_undo_snapshot(state, wcb)


def _remember_rejected_current_draft(state, wcb, remove_from_ui=True):
    return drafts.remember_rejected_current_draft(state, wcb, remove_from_ui=remove_from_ui)


def _revision_feedback_message(command):
    return drafts.revision_feedback_message(command)


def _commit_current_draft(state, wcb=None):
    return drafts.commit_current_draft(state, wcb)


def _replace_revision_ui_with_accepted(wcb, commit_info, title="事件已保存"):
    return drafts.replace_revision_ui_with_accepted(wcb, commit_info, title=title)
