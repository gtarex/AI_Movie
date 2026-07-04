import re

from .state import UNDO_STACK_LIMIT
from .storage import read_file_raw
from .text_utils import story_message_box


class SceneRecordService:
    def scene_entry_start(self, raw, heading_start):
        index = heading_start
        while index > 0 and raw[index - 1].isspace():
            index -= 1
        return index

    def scene_record_blocks(self, raw):
        pattern = re.compile(
            r"(?m)^\s*\*\*(DIRECTOR|EVENT|GOD COMMAND|CORRECTION|SCENE SYNOPSIS):\*\*\s*$"
            r"|^\s*\[(Scene Summary)\]\s*$"
        )
        matches = list(pattern.finditer(raw or ""))
        if not matches:
            content = (raw or "").strip()
            return [{"label": "EVENT", "content": content, "start": 0}] if content else []

        blocks = []
        prefix = (raw[:matches[0].start()] or "").strip()
        if prefix:
            blocks.append({"label": "EVENT", "content": prefix, "start": 0})
        for index, match in enumerate(matches):
            label = match.group(1) or match.group(2)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            content = raw[match.end():end].strip()
            blocks.append({"label": label, "content": content, "start": match.start()})
        return blocks

    def parse_scene_record(self, scene_path, scene_name):
        raw = read_file_raw(scene_path)
        blocks = self.scene_record_blocks(raw)
        ui_messages = [{"role": "assistant", "content": f"**{scene_name}**\n已载入场景记录，可直接继续；输入 /k 或评价即可推进。"}]
        history = []
        undo_snapshots = []
        first_story = ""
        summaries = []
        pending_user = None

        def remember_snapshot(scene_before, history_before, ui_start):
            undo_snapshots.append({
                "scene_path": scene_path,
                "scene_before": scene_before,
                "writing_history_before": [dict(message) for message in history_before],
                "ui_start": ui_start,
            })

        for block in blocks:
            label = block.get("label", "")
            content = (block.get("content") or "").strip()
            if not content:
                continue
            entry_start = self.scene_entry_start(raw, block.get("start", 0))

            if label == "Scene Summary":
                summaries.append(content)
                continue

            if label in ("DIRECTOR", "GOD COMMAND"):
                user_content = content
                if label == "GOD COMMAND" and not content.lower().startswith("/u"):
                    user_content = f"[上帝指令]\n{content}"
                pending_user = {
                    "scene_before": raw[:entry_start].rstrip(),
                    "history_before": [dict(message) for message in history],
                    "ui_start": len(ui_messages),
                }
                history.append({"role": "user", "content": user_content})
                ui_messages.append({"role": "user", "content": story_message_box(content, status="draft", title="导演输入")})
                continue

            if label in ("EVENT", "CORRECTION", "SCENE SYNOPSIS"):
                if not first_story:
                    first_story = content
                if pending_user:
                    remember_snapshot(
                        pending_user["scene_before"],
                        pending_user["history_before"],
                        pending_user["ui_start"],
                    )
                else:
                    remember_snapshot(raw[:entry_start].rstrip(), history, len(ui_messages))
                history.append({"role": "assistant", "content": content})
                ui_messages.append({"role": "assistant", "content": story_message_box(content, status="accepted")})
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


scene_records = SceneRecordService()


def _scene_entry_start(raw, heading_start):
    return scene_records.scene_entry_start(raw, heading_start)


def _scene_record_blocks(raw):
    return scene_records.scene_record_blocks(raw)


def _parse_scene_record(scene_path, scene_name):
    return scene_records.parse_scene_record(scene_path, scene_name)
