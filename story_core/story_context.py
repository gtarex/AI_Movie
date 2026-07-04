import os
import re

from config import MAX_TOKENS_SCENE_MEMORY, TOKEN_MULTIPLIER_PER_CHAR
from .api import API_TIMEOUT_SUPPORT, call_deepseek
from .characters import extract_character_name
from .state import MAX_FACT_REWRITE_ATTEMPTS, STORY_HISTORY_CONTEXT_CHARS, turn_label
from .storage import character_path, current_story_dir, outline_path, read_file, read_file_raw, write_file
from .text_utils import limit_chars


CONTEXT_WARN_TOKENS = 500000


class StoryContextService:
    def writing_history_chars(self, history):
        return sum(len(str(msg.get("content", ""))) for msg in (history or []))

    def history_paragraph_units(self, history):
        units = []
        for msg in history or []:
            role = msg.get("role") if msg.get("role") in ("user", "assistant") else "user"
            content = str(msg.get("content", "")).replace("\r\n", "\n").replace("\r", "\n")
            for paragraph in content.split("\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    units.append({"role": role, "content": paragraph})
        return units

    def rebuild_history_from_paragraph_units(self, units):
        rebuilt = []
        for unit in units or []:
            role = unit.get("role") if unit.get("role") in ("user", "assistant") else "user"
            content = str(unit.get("content", "")).strip()
            if not content:
                continue
            if rebuilt and rebuilt[-1].get("role") == role:
                rebuilt[-1]["content"] = f"{rebuilt[-1]['content']}\n{content}"
            else:
                rebuilt.append({"role": role, "content": content})
        return rebuilt

    def recent_writing_history_window(self, state):
        history = [dict(m) for m in (state.get("writing_history", []) or []) if str(m.get("content", "")).strip()]
        selected = []
        used_chars = 0
        for unit in reversed(self.history_paragraph_units(history)):
            content_len = len(str(unit.get("content", "")))
            if selected and used_chars + content_len > STORY_HISTORY_CONTEXT_CHARS:
                break
            selected.append(unit)
            used_chars += content_len
        return self.rebuild_history_from_paragraph_units(reversed(selected))

    def story_history_message_for_api(self, msg):
        role = msg.get("role") if msg.get("role") in ("user", "assistant") else "user"
        label = "导演" if role == "user" else "剧情"
        content = str(msg.get("content", "")).strip()
        return {"role": role, "content": f"[{label}]\n{content}"}

    def api_writing_context(self, state):
        return [self.story_history_message_for_api(m) for m in self.recent_writing_history_window(state)]

    def recent_history_context_text(self, state):
        lines = []
        for index, msg in enumerate(self.recent_writing_history_window(state), 1):
            role = msg.get("role")
            label = "导演" if role == "user" else "剧情"
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{index}. {label}: {content}")
        return "\n".join(lines)

    def revision_history_context_text(self, revision_history):
        lines = []
        for index, msg in enumerate(revision_history or [], 1):
            role = msg.get("role")
            label = "导演修订" if role == "user" else "上一版草稿"
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{index}. {label}: {content}")
        return "\n".join(lines)

    def god_instruction_text(self, user_prompt):
        text = str(user_prompt or "").strip()
        if text.startswith("[上帝指令]"):
            text = text[len("[上帝指令]"):].strip()
        if text.lower().startswith("/u"):
            text = text[2:].strip()
        return f"/u {text}".strip() if text else "/u"

    def god_context_text(self, state, revision_history=None):
        parts = []
        scenario = self.story_scenario_context(state).strip()
        if scenario:
            parts.append(f"[当前场景设定]\n{scenario}")
        outline = self.past_outline(state).strip()
        if outline:
            parts.append(f"[章节提纲]\n{outline}")
        if state.get("scene_summary"):
            parts.append(f"[当前场景总结 / 压缩前略]\n{state.get('scene_summary')}")
        recent_history = self.recent_history_context_text(state)
        if recent_history:
            parts.append(f"[最近正式剧情历史 / 未压缩现场]\n{recent_history}")
        revisions = self.revision_history_context_text(revision_history)
        if revisions:
            parts.append(f"[未接受草稿修订链]\n{revisions}")
        return "\n\n".join(parts)

    def build_god_api_messages(self, state, user_prompt, revision_history=None):
        character_context = self.build_character_context(state, active_only=True).strip()
        if not character_context:
            character_context = self.build_character_context(state, active_only=False).strip()
        return self.clean_story_messages([
            {"role": "system", "content": "你是小说写手。"},
            {"role": "user", "content": self.god_instruction_text(user_prompt)},
            {"role": "system", "content": f"下面是世界观设定：\n{state.get('world_bg', '').strip() or '无'}"},
            {"role": "system", "content": f"下面是人物卡：\n{character_context or '无'}"},
            {"role": "system", "content": f"下面是上下文：\n{self.god_context_text(state, revision_history) or '无'}"},
        ])

    def past_outline(self, state):
        return read_file(outline_path(current_story_dir(state)))

    def outline_context_block(self, state):
        outline = self.past_outline(state)
        return f"\n\n[过去章节提纲]\n{outline}" if outline else ""

    def history_message_count(self, state):
        return len(state.get("writing_history", []) or [])

    def accepted_event_count(self, state):
        return sum(1 for m in state.get("writing_history", []) if m.get("role") == "assistant")

    def reset_scene_support_context(self, state):
        state["scene_summary"] = ""
        state["scene_summary_history_count"] = 0
        state["scene_summary_history_chars"] = 0
        state["fact_pack"] = {}
        state["fact_pack_history_count"] = -1
        state["fact_pack_scene"] = ""
        state["accepted_event_total"] = 0
        state["_last_fact_check_issues"] = []
        return state

    def character_field(self, profile, field):
        match = re.search(rf"^{re.escape(field)}\s*[:：]\s*(.*)$", profile or "", re.MULTILINE)
        return match.group(1).strip() if match else ""

    def context_character_codenames(self, state, active_only=True):
        story_dir = current_story_dir(state)
        chars = state.get("characters", {}) or {}
        active = state.get("active_characters", []) or []
        if active_only and active:
            return list(active)
        target_codenames = list(active) if active else sorted(chars.keys())
        if not target_codenames:
            char_dir = os.path.join(story_dir, "char")
            if os.path.isdir(char_dir):
                target_codenames = sorted(
                    filename[:-4] for filename in os.listdir(char_dir)
                    if filename.endswith(".txt") and not filename.startswith(".")
                )
        return target_codenames

    def character_key_fact_lines(self, state):
        chars = state.get("characters", {}) or {}
        targets = state.get("active_characters", []) or sorted(chars.keys())
        lines = []
        story_dir = current_story_dir(state)
        for codename in targets:
            profile = read_file(character_path(story_dir, codename)) or chars.get(codename, "")
            if not profile:
                continue
            name = extract_character_name(profile) or codename
            parts = [f"{name}（{codename}）"]
            for field in ["性别", "年龄", "职业", "人际关系", "能力"]:
                value = self.character_field(profile, field)
                if value:
                    parts.append(f"{field}：{value}")
            lines.append("；".join(parts))
        return lines

    def local_fact_pack(self, state):
        immutable = self.character_key_fact_lines(state)
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

    def clean_fact_pack(self, data, fallback):
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

    def fact_pack_text(self, state):
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

    def history_text(self, state):
        lines = []
        for index, msg in enumerate(state.get("writing_history", []) or [], 1):
            label = "导演" if msg.get("role") == "user" else "剧情"
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{index}. {label}: {content}")
        return "\n".join(lines)

    def fallback_full_scene_summary(self, state, max_chars=1200):
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

    def compact_writing_history_after_summary(self, state):
        history = state.get("writing_history", []) or []
        total_chars = self.writing_history_chars(history)
        if total_chars <= STORY_HISTORY_CONTEXT_CHARS:
            return 0
        target_keep_chars = max(total_chars // 3, 1)
        keep = []
        kept_chars = 0
        units = self.history_paragraph_units(history)
        for unit in reversed(units):
            keep.append(unit)
            kept_chars += len(str(unit.get("content", "")))
            if kept_chars >= target_keep_chars:
                break
        keep = list(reversed(keep))
        remove_count = len(units) - len(keep)
        if remove_count <= 0:
            return 0
        state["writing_history"] = self.rebuild_history_from_paragraph_units(keep)
        return remove_count

    def update_scene_summary_if_needed(self, state, client, force=False):
        history = state.get("writing_history", []) or []
        count = int(state.get("accepted_event_total") or self.accepted_event_count(state))
        history_chars = self.writing_history_chars(history)
        if not history:
            state["scene_summary"] = ""
            state["scene_summary_history_count"] = 0
            state["scene_summary_history_chars"] = 0
            return False
        if not force and history_chars <= STORY_HISTORY_CONTEXT_CHARS:
            return False
        if (
            not force
            and state.get("scene_summary")
            and state.get("scene_summary_history_count") == count
            and state.get("scene_summary_history_chars") == history_chars
        ):
            return False
        fallback = self.fallback_full_scene_summary(state)
        if client is None:
            state["scene_summary"] = fallback
            state["scene_summary_history_count"] = count
            state["scene_summary_history_chars"] = history_chars
            self.compact_writing_history_after_summary(state)
            return bool(fallback)
        system_prompt = (
            "你是互动小说当前场景的连续性整理器。请根据正式通过的本幕记录，写一个当前场景总结。\n"
            "要求：覆盖从开场到现在的所有已发生事件；按时间顺序简述；保留角色状态、地点、目标、矛盾、已造成的后果；"
            "对最近仍会保留在上下文中的内容只做简述，不要复写细节；不要评价文风，不要续写剧情。"
        )
        user_prompt = (
            f"[本幕场景]\n{state.get('scene_premise','')}\n\n"
            f"[已有场景总结]\n{state.get('scene_summary','') or '无'}\n\n"
            f"[正式通过记录]\n{self.history_text(state)}"
        )
        raw, err = call_deepseek(client, state["model"], [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], max_tok=MAX_TOKENS_SCENE_MEMORY, timeout=API_TIMEOUT_SUPPORT, thinking_type="disabled")
        summary = limit_chars(raw or fallback, 1200)
        if summary:
            state["scene_summary"] = summary
            state["scene_summary_history_count"] = count
            state["scene_summary_history_chars"] = history_chars
            self.compact_writing_history_after_summary(state)
            return True
        return False

    def refresh_fact_pack(self, state, client, force=False):
        count = int(state.get("accepted_event_total") or self.accepted_event_count(state))
        scene_name = state.get("scene_name", "")
        fallback = self.local_fact_pack(state)
        existing_pack = state.get("fact_pack") if isinstance(state.get("fact_pack"), dict) else {}
        pack = self.clean_fact_pack(existing_pack, fallback) if existing_pack else fallback
        state["fact_pack"] = pack
        state["fact_pack_history_count"] = count
        state["fact_pack_scene"] = scene_name
        return pack

    def refresh_scene_support_context(self, state, client, force=False):
        self.update_scene_summary_if_needed(state, client, force=force)
        self.refresh_fact_pack(state, client, force=force)
        return state

    def fact_check_draft(self, state, client, draft):
        return True, [], ""

    def validate_and_rewrite_draft(self, state, client, draft, original_msgs, max_tok, attempts=MAX_FACT_REWRITE_ATTEMPTS):
        return draft

    def fallback_scene_summary_from_history(self, state, max_chars=300):
        events = [m.get("content", "").strip() for m in state.get("writing_history", []) if m.get("role") == "assistant" and m.get("content")]
        if not events:
            return ""
        joined = " / ".join(limit_chars(event.replace("\n", " "), 120) for event in events[-3:])
        return limit_chars(joined, max_chars)

    def update_outline_file(self, state, scene_name, summary):
        summary = limit_chars(" ".join((summary or "").split()), 500)
        if not summary:
            return False
        story_dir = current_story_dir(state)
        path = outline_path(story_dir)
        key = os.path.splitext(scene_name or state.get("scene_name", "scene_01.txt"))[0]
        line = f"{key}: {summary}"
        lines = [existing.rstrip() for existing in read_file_raw(path).splitlines() if existing.strip()]
        replaced = False
        for index, existing in enumerate(lines):
            if existing.startswith(f"{key}:"):
                lines[index] = line
                replaced = True
                break
        if not replaced:
            lines.append(line)
        write_file(path, "\n".join(lines))
        return True

    def build_character_context(self, state, active_only=True):
        story_dir = current_story_dir(state)
        chars = state.get("characters", {})
        target_codenames = self.context_character_codenames(state, active_only=active_only)
        if not target_codenames:
            return ""

        parts = []
        for codename in target_codenames:
            profile = read_file(character_path(story_dir, codename))
            if not profile:
                profile = chars.get(codename, "")
            if profile:
                parts.append(f"\n--- {codename} ---\n{profile}\n")
                state["characters"][codename] = profile
            else:
                parts.append(f"\n--- {codename} ---\n（角色档案未找到）\n")
        return "".join(parts)

    def build_character_personality_context(self, state, active_only=True):
        story_dir = current_story_dir(state)
        chars = state.get("characters", {}) or {}
        lines = []
        for codename in self.context_character_codenames(state, active_only=active_only):
            profile = read_file(character_path(story_dir, codename)) or chars.get(codename, "")
            if not profile:
                continue
            name = extract_character_name(profile) or codename
            fields = []
            for field in ["性格特征", "爱好", "人际关系", "能力", "过往经历"]:
                value = self.character_field(profile, field)
                if value:
                    fields.append(f"{field}：{limit_chars(value, 220)}")
            if fields:
                lines.append(f"{name}（{codename}）：{'；'.join(fields)}")
        return "\n".join(lines)

    def story_section_message(self, identifier, title, content, role="system"):
        content = str(content or "").strip()
        if not content:
            return None
        return {"role": role, "content": f"[{title}]\n{content}", "_id": identifier}

    def clean_story_messages(self, messages):
        cleaned = []
        for msg in messages:
            if not msg:
                continue
            item = {key: value for key, value in msg.items() if not key.startswith("_")}
            if str(item.get("content", "")).strip():
                cleaned.append(item)
        return cleaned

    def story_world_info_before(self, state):
        parts = []
        if state.get("world_bg"):
            parts.append(f"[World.txt]\n{state.get('world_bg')}")
        past_outline = self.past_outline(state)
        if past_outline:
            parts.append(f"[outline.txt]\n{past_outline}")
        return "\n\n".join(parts)

    def story_world_info_after(self, state):
        parts = []
        fact_text = self.fact_pack_text(state)
        if fact_text:
            parts.append(f"[事实包]\n{fact_text}")
        if state.get("scene_summary"):
            parts.append(f"[当前场景总结]\n{state.get('scene_summary')}")
        return "\n\n".join(parts)

    def story_scenario_context(self, state):
        parts = []
        premise = state.get("scene_premise", "")
        outline = state.get("scene_outline", "")
        if premise:
            parts.append(premise)
        if outline and outline.strip() != premise.strip():
            parts.append(f"[场景构架]\n{outline}")
        if not parts:
            parts.append("无大纲开始：以导演当前输入作为开场锚点，不补造未给出的过去场景。")
        return "\n\n".join(parts)

    def story_main_prompt(self, purpose="event"):
        if purpose == "god":
            return (
                "你是互动小说的最高优先级修正器。用户会用 /u 发出上帝指令，用来修正剧情错误、连续性问题、角色误写或世界观冲突。"
                "必须服从上帝指令，并输出可直接写入正式故事记录的修正版剧情/补丁。"
            )
        if purpose == "synopsis":
            return "你是互动小说写手。导演使用 /s 命令，需要你从当前剧情点出发，用大纲完成当前场景。"
        return "你是互动小说写手。用户是故事导演；请依据正式历史、当前设定和导演输入，用中文继续当前事件。"

    def story_control_prompt(self, state, purpose="event"):
        common = [
            "只依据正式历史、当前设定、人物卡和导演本轮输入。",
            "场景总结是压缩记忆；最近正式剧情历史是当前现场的直接承接点。",
            "章节提纲只用于连续性，禁止复述或重演旧章节。",
            "保持角色年龄、身份、关系、能力和知识范围一致。",
            "只输出正文内容，不写戏外解释、元评论、系统规则或导演原话。",
        ]
        if purpose == "god":
            rules = [
                "上帝指令优先级最高；按指令修正剧情错误。",
                "不要消耗任何角色回合，不要让当前角色额外行动，除非上帝指令明确要求。",
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
                "聚焦当前行动、对话、环境变化和角色反应。",
                "如果导演给出长期目标，本段只写下一个触发、阻碍或进展；明确要求立刻完成时除外。",
            ]
        return "\n".join(f"- {item}" for item in rules + common)

    def story_context_messages(self, state, purpose="event", include_control=False):
        character_context = self.build_character_context(state, active_only=True)
        if not character_context:
            character_context = self.build_character_context(state, active_only=False)
        if purpose in ("event", "synopsis"):
            continuity = []
            outline = self.past_outline(state)
            if outline:
                continuity.append(f"[章节提纲]\n{outline}")
            if state.get("scene_summary"):
                continuity.append(f"[当前场景总结 / 压缩前略]\n{state.get('scene_summary')}")
            recent_history = self.recent_history_context_text(state)
            if recent_history:
                continuity.append(f"[最近正式剧情历史 / 未压缩现场]\n{recent_history}")
            messages = [
                self.story_section_message("main", "主提示", self.story_main_prompt(purpose)),
                self.story_section_message("worldInfo", "世界观设定", state.get("world_bg", "")),
                self.story_section_message("charDescription", "人物卡", character_context.strip() if character_context else ""),
                self.story_section_message("scenario", "当前场景", self.story_scenario_context(state)),
                self.story_section_message("continuity", "连续性上下文", "\n\n".join(continuity)),
            ]
            if include_control:
                messages.append(self.story_section_message("controlPrompts", "输出约束", self.story_control_prompt(state, purpose)))
            return self.clean_story_messages(messages)
        messages = [
            self.story_section_message("worldInfoBefore", "世界信息 / 过去提纲", self.story_world_info_before(state)),
            self.story_section_message("main", "主提示", self.story_main_prompt(purpose)),
            self.story_section_message("worldInfoAfter", "当前记忆 / 事实约束", self.story_world_info_after(state)),
            self.story_section_message("charDescription", "角色卡", character_context.strip() if character_context else ""),
            self.story_section_message("scenario", "本幕场景", self.story_scenario_context(state)),
        ]
        if include_control:
            messages.append(self.story_section_message("controlPrompts", "输出约束", self.story_control_prompt(state, purpose)))
        return self.clean_story_messages(messages)

    def story_control_message(self, state, purpose="event"):
        return self.clean_story_messages([self.story_section_message("controlPrompts", "输出约束", self.story_control_prompt(state, purpose))])[0]

    def build_story_api_messages(self, state, user_prompt, purpose="event", revision_history=None):
        if purpose == "god":
            return self.build_god_api_messages(state, user_prompt, revision_history=revision_history)
        messages = self.story_context_messages(state, purpose=purpose)
        if purpose not in ("event", "synopsis"):
            messages.extend(self.api_writing_context(state))
        if revision_history:
            messages.extend([dict(message) for message in revision_history])
        messages.append(self.story_control_message(state, purpose=purpose))
        if user_prompt is not None:
            messages.append({"role": "user", "content": str(user_prompt)})
        return messages

    def build_wsp(self, state):
        return "\n\n".join(message["content"] for message in self.story_context_messages(state, purpose="event", include_control=True))

    def build_event_prompt(self, state):
        return "自行推演后续剧情。"

    def estimate_context_tokens(self, state):
        writing_history = self.api_writing_context(state)
        rejected = state.get("rejected_draft_history", [])
        total_chars = sum(len(message.get("content", "")) for message in writing_history + rejected)
        total_chars += len(state.get("world_bg", "")) + len(state.get("scene_premise", ""))
        return int(total_chars * TOKEN_MULTIPLIER_PER_CHAR)

    def check_context_size(self, state, wcb):
        estimated = self.estimate_context_tokens(state)
        if estimated > CONTEXT_WARN_TOKENS:
            wcb.append({"role": "assistant",
                "content": f"⚠️ **上下文警告**：当前幕已累积约 {estimated // 1000}K tokens（上限 {CONTEXT_WARN_TOKENS // 1000}K），"
                           f"建议尽快「结束本幕」以重置上下文，避免超出 AI 处理上限。"})
            return True
        if estimated > CONTEXT_WARN_TOKENS * 0.7:
            wcb.append({"role": "assistant",
                "content": f"⚡ **上下文提醒**：当前幕已累积约 {estimated // 1000}K tokens，接近 {CONTEXT_WARN_TOKENS // 1000}K 上限。"
                           f"建议在适当时候「结束本幕」。**（本条提醒仅显示一次，之后不再重复）**"})
            return True
        return False

    def build_skip_synopsis_prompt(self, state):
        return "\n\n".join(message["content"] for message in self.story_context_messages(state, purpose="synopsis", include_control=True))

    def build_god_command_prompt(self, state):
        user_prompt = state.get("pending_turn_input") or state.get("pending_user_input") or "/u"
        return "\n\n".join(
            message["content"]
            for message in self.build_god_api_messages(state, user_prompt, state.get("rejected_draft_history", []))
        )


story_context = StoryContextService()


def _recent_writing_history_window(state):
    return story_context.recent_writing_history_window(state)


def _story_history_message_for_api(msg):
    return story_context.story_history_message_for_api(msg)


def _api_writing_context(state):
    return story_context.api_writing_context(state)


def _past_outline(state):
    return story_context.past_outline(state)


def _outline_context_block(state):
    return story_context.outline_context_block(state)


def _history_message_count(state):
    return story_context.history_message_count(state)


def _accepted_event_count(state):
    return story_context.accepted_event_count(state)


def _reset_scene_support_context(state):
    return story_context.reset_scene_support_context(state)


def _character_field(profile, field):
    return story_context.character_field(profile, field)


def _character_key_fact_lines(state):
    return story_context.character_key_fact_lines(state)


def _local_fact_pack(state):
    return story_context.local_fact_pack(state)


def _clean_fact_pack(data, fallback):
    return story_context.clean_fact_pack(data, fallback)


def _fact_pack_text(state):
    return story_context.fact_pack_text(state)


def _history_text(state):
    return story_context.history_text(state)


def _fallback_full_scene_summary(state, max_chars=1200):
    return story_context.fallback_full_scene_summary(state, max_chars)


def _compact_writing_history_after_summary(state):
    return story_context.compact_writing_history_after_summary(state)


def _update_scene_summary_if_needed(state, client, force=False):
    return story_context.update_scene_summary_if_needed(state, client, force=force)


def _refresh_fact_pack(state, client, force=False):
    return story_context.refresh_fact_pack(state, client, force=force)


def _refresh_scene_support_context(state, client, force=False):
    return story_context.refresh_scene_support_context(state, client, force=force)


def _fact_check_draft(state, client, draft):
    return story_context.fact_check_draft(state, client, draft)


def _validate_and_rewrite_draft(state, client, draft, original_msgs, max_tok, attempts=MAX_FACT_REWRITE_ATTEMPTS):
    return story_context.validate_and_rewrite_draft(state, client, draft, original_msgs, max_tok, attempts=attempts)


def _fallback_scene_summary_from_history(state, max_chars=300):
    return story_context.fallback_scene_summary_from_history(state, max_chars)


def _update_outline_file(state, scene_name, summary):
    return story_context.update_outline_file(state, scene_name, summary)


def _context_character_codenames(state, active_only=True):
    return story_context.context_character_codenames(state, active_only=active_only)


def _build_character_context(state, active_only=True):
    return story_context.build_character_context(state, active_only=active_only)


def _build_character_personality_context(state, active_only=True):
    return story_context.build_character_personality_context(state, active_only=active_only)


def _story_section_message(identifier, title, content, role="system"):
    return story_context.story_section_message(identifier, title, content, role=role)


def _clean_story_messages(messages):
    return story_context.clean_story_messages(messages)


def _story_world_info_before(state):
    return story_context.story_world_info_before(state)


def _story_world_info_after(state):
    return story_context.story_world_info_after(state)


def _story_scenario_context(state):
    return story_context.story_scenario_context(state)


def _story_main_prompt(purpose="event"):
    return story_context.story_main_prompt(purpose)


def _story_control_prompt(state, purpose="event"):
    return story_context.story_control_prompt(state, purpose)


def _story_context_messages(state, purpose="event", include_control=False):
    return story_context.story_context_messages(state, purpose=purpose, include_control=include_control)


def _story_control_message(state, purpose="event"):
    return story_context.story_control_message(state, purpose=purpose)


def _build_story_api_messages(state, user_prompt, purpose="event", revision_history=None):
    return story_context.build_story_api_messages(state, user_prompt, purpose=purpose, revision_history=revision_history)


def _build_wsp(state):
    return story_context.build_wsp(state)


def _build_event_prompt(state):
    return story_context.build_event_prompt(state)


def _estimate_context_tokens(state):
    return story_context.estimate_context_tokens(state)


def _check_context_size(state, wcb):
    return story_context.check_context_size(state, wcb)


def _build_skip_synopsis_prompt(state):
    return story_context.build_skip_synopsis_prompt(state)


def _build_god_command_prompt(state):
    return story_context.build_god_command_prompt(state)
