import os
import re

import gradio as gr

from .api import call_deepseek, get_client
from .characters import sanitize_codename
from .setup_handlers import load_characters_from_char_dir
from .storage import character_path, current_story_dir, ensure_char_dir, read_file, write_file
from .story_context import _build_character_context
from .text_utils import limit_chars


class BuildHandlers:
    def handle_world_generate_or_revise(self, state, concept, world_display):
        story_dir = current_story_dir(state)
        if not concept or not concept.strip():
            return state, world_display, "请输入世界设定或修改意见。"
        client = get_client(state)
        if client is None:
            return state, world_display, "请先在「启动 / API」里输入 DeepSeek API Key。"
        world_bg = state.get("world_bg", "")
        if world_bg:
            msgs = [
                {"role": "system", "content": "你是专业小说世界观设计师。根据反馈修改已有世界观，必须控制在 200 个中文字符以内。请用中文直接输出，不要寒暄。"},
                {"role": "assistant", "content": world_bg},
                {"role": "user", "content": f"请根据以下意见修改世界观，结果控制在 200 中文字符以内：\n\n{concept.strip()}"},
            ]
            result, err = call_deepseek(client, state["model"], msgs, max_tok=400)
            if err:
                return state, world_display, f"错误：{err}"
            result = limit_chars(result, 200)
            state["_wm"] = msgs + [{"role": "assistant", "content": result}]
            return state, result, "已根据意见修改。请审阅后点击「接受/保存世界观」。"

        msgs = [
            {"role": "system", "content": "你是专业小说世界观设计师。请用中文直接输出结构清晰的世界背景，不要寒暄。必须控制在 200 个中文字符以内。"},
            {"role": "user", "content": concept.strip()},
        ]
        result, err = call_deepseek(client, state["model"], msgs, max_tok=400)
        if err:
            return state, world_display, f"错误：{err}"
        result = limit_chars(result, 200)
        state["_wm"] = msgs + [{"role": "assistant", "content": result}]
        return state, result, "已生成。请审阅后点击「接受/保存世界观」。"

    def handle_save_world(self, state, world_display):
        if not world_display or not world_display.strip():
            return state, world_display, "没有可保存的世界观。"
        story_dir = current_story_dir(state)
        world_display = limit_chars(world_display, 200)
        write_file(os.path.join(story_dir, "World.txt"), world_display)
        state["world_bg"] = world_display
        return state, world_display, f"已保存到 {story_dir}/World.txt（已确保 200 字以内）"

    def handle_generate_character(self, state, char_desc, char_display, char_name):
        if char_desc and char_desc.strip().lower() == "/k":
            choices = load_characters_from_char_dir(state)
            msg = f"已跳过角色生成，并载入 char 文件夹中的 {len(choices)} 个角色。"
            state.pop("_editing_character_id", None)
            state.pop("_pending_character_id", None)
            return state, char_display, char_name, msg, gr.update(choices=choices, value=None)
        if not char_desc or not char_desc.strip():
            return state, char_display, char_name, "请输入角色提示词；或输入 /k 直接使用当前 char 文件夹中的角色。", gr.update()
        client = get_client(state)
        if client is None:
            return state, char_display, char_name, "请先在「启动 / API」里输入 DeepSeek API Key。", gr.update()
        story_dir = current_story_dir(state)
        world_bg = state.get("world_bg", "")
        editing_id = state.get("_editing_character_id", "")
        if editing_id:
            existing_chars = _build_character_context(state, active_only=False)
            codename = editing_id
            sys_msg = (
                "你正在修改一个已有角色档案。必须严格保持字段格式：姓名、性别、年龄、职业、外貌、性格特征、爱好、过往经历、人际关系、能力。\n"
                f"姓名行里的 ID 必须保持为 {codename}。角色档案控制在 200 words 以内。"
            )
            if world_bg:
                sys_msg += f"\n\n世界观：\n{world_bg}"
            if existing_chars:
                sys_msg += f"\n\n[全部已有角色 —— 必须避免与之矛盾]\n{existing_chars}"
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "assistant", "content": char_display.strip()},
                {"role": "user", "content": f"请根据以下意见修改角色档案，保持字段格式，ID 为 {codename}：\n\n{char_desc.strip()}"},
            ]
            result, err = call_deepseek(client, state["model"], msgs, max_tok=700)
            if err:
                return state, char_display, char_name, f"错误：{err}", gr.update()
            state["_cm"] = msgs + [{"role": "assistant", "content": result}]
            return state, result, char_name, f"已根据意见修改角色：{codename}", gr.update()

        existing_chars = _build_character_context(state, active_only=False)
        is_batch = bool(re.search(r"(多个角色|以下几个|以下角色|生成.*个|批量|一群人|几个角色|小队|两个|三个|四个|五个|六个|一群|一批)", char_desc.strip()))
        system_content = (
            "你是专业小说角色设计师。根据用户的提示词，生成一个或多个角色档案。\n"
            "用中文直接输出，不要寒暄。多角色之间用 === 分隔。\n"
            "每个角色档案必须严格使用下面字段和顺序：\n"
            "姓名：角色中文名（charN）\n性别：\n年龄：\n职业：\n外貌：\n\n"
            "性格特征：\n爱好：\n过往经历：\n人际关系：\n能力：\n\n"
            "姓名行括号里的 ID 统一使用 char1, char2, char3... 依次编号。\n"
            "性格特征和爱好写几个关键词；能力字段必须填写：普通人写「无」，有战斗/魔法/超能力的角色写具体能力描述。\n"
            "每个角色档案控制在 200 words 以内。角色之间不得互相矛盾。"
        )
        if world_bg:
            system_content += f"\n\n世界观：\n{world_bg}"
        if existing_chars:
            system_content += f"\n\n[已有角色 —— 必须避免与之矛盾]\n{existing_chars}"
        max_tok = 2500 if is_batch else 700
        result, err = call_deepseek(client, state["model"], [
            {"role": "system", "content": system_content},
            {"role": "user", "content": char_desc.strip()},
        ], max_tok=max_tok)
        if err:
            return state, char_display, char_name, f"错误：{err}", gr.update()
        state["_cm"] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": char_desc.strip()},
            {"role": "assistant", "content": result},
        ]
        new_name = self.next_char_name(state)
        state["_editing_character_id"] = ""
        state["_pending_character_id"] = ""
        msg = "已生成角色。请审阅、设置文件名并保存。"
        if "===" in result:
            msg = "已生成多个角色（=== 分隔）。可分别编辑后依次保存，或点击「批量保存」一次性保存。"
        return state, result, new_name, msg, gr.update()

    def next_char_name(self, state):
        story_dir = current_story_dir(state)
        char_dir = ensure_char_dir(story_dir)
        existing = set()
        for filename in os.listdir(char_dir):
            if filename.endswith(".txt") and not filename.startswith("."):
                existing.add(os.path.splitext(filename)[0])
        number = 1
        while f"char{number}" in existing:
            number += 1
        return f"char{number}"

    def handle_load_character(self, state, selected, char_name):
        if not selected:
            return state, "", char_name, "请选择角色，或直接生成新角色。"
        codename = selected if isinstance(selected, str) else selected[0] if isinstance(selected, list) else str(selected)
        story_dir = current_story_dir(state)
        path = character_path(story_dir, codename)
        profile = read_file(path)
        if not profile:
            return state, "", char_name, f"找不到角色文件：{path}"
        state.setdefault("characters", {})[codename] = profile
        state["_editing_character_id"] = codename
        state["_pending_character_id"] = ""
        world_bg = state.get("world_bg", "")
        existing_chars = _build_character_context(state, active_only=False)
        sys_msg = (
            "你正在修改一个已有角色档案。必须严格保持字段格式：姓名、性别、年龄、职业、外貌、性格特征、爱好、过往经历、人际关系、能力。"
            f"姓名行里的 ID 必须保持为 {codename}。角色档案必须控制在 200 words 以内。"
        )
        if world_bg:
            sys_msg += f"\n\n世界观：\n{world_bg}"
        if existing_chars:
            sys_msg += f"\n\n[全部已有角色 —— 必须避免与之矛盾]\n{existing_chars}"
        state["_cm"] = [
            {"role": "system", "content": sys_msg},
            {"role": "assistant", "content": profile},
        ]
        return state, profile, codename, f"已载入角色：{codename}"

    def handle_save_character(self, state, char_name, char_display):
        if not char_display or not char_display.strip():
            return state, char_name, char_display, "没有可保存的角色。", gr.update()
        story_dir = current_story_dir(state)
        ensure_char_dir(story_dir)
        if not char_name or not char_name.strip():
            char_name = self.next_char_name(state)
        codename = sanitize_codename(char_name.strip())
        if not codename:
            codename = self.next_char_name(state)
        path = character_path(story_dir, codename)
        write_file(path, char_display.strip())
        state["characters"][codename] = char_display.strip()
        state["_editing_character_id"] = codename
        state["_pending_character_id"] = ""
        choices = load_characters_from_char_dir(state)
        return state, codename, char_display, f"已保存角色：{codename}", gr.update(choices=choices, value=codename)

    def handle_delete_character(self, state, selected):
        if not selected:
            return state, gr.update(), gr.update(), "请先选择角色。"
        codename = selected if isinstance(selected, str) else selected[0] if isinstance(selected, list) else str(selected)
        story_dir = current_story_dir(state)
        path = character_path(story_dir, codename)
        if os.path.exists(path):
            os.remove(path)
        if codename in state["characters"]:
            del state["characters"][codename]
        if state.get("_editing_character_id") == codename:
            state["_editing_character_id"] = ""
        if state.get("_pending_character_id") == codename:
            state["_pending_character_id"] = ""
        choices = list(state["characters"].keys())
        return state, gr.update(choices=choices, value=None), "", f"已删除角色文件：{codename}"

    def handle_rename_character(self, state, new_name):
        old_codename = state.get("_editing_character_id", "")
        if not old_codename:
            return state, old_codename, "请先载入一个角色再重命名。", gr.update()
        if not new_name or not new_name.strip():
            return state, old_codename, "请输入新文件名。", gr.update()
        new_codename = sanitize_codename(new_name.strip())
        if not new_codename:
            return state, old_codename, "文件名无效。", gr.update()
        if old_codename == new_codename:
            return state, old_codename, f"文件名未变化：{old_codename}", gr.update()
        story_dir = current_story_dir(state)
        old_path = character_path(story_dir, old_codename)
        new_path = character_path(story_dir, new_codename)
        if not os.path.exists(old_path):
            return state, old_codename, f"找不到角色文件：{old_codename}", gr.update()
        if os.path.exists(new_path):
            return state, old_codename, f"目标文件名 {new_codename} 已存在。", gr.update()
        os.rename(old_path, new_path)
        if old_codename in state.get("characters", {}):
            state["characters"][new_codename] = state["characters"].pop(old_codename)
        state["_editing_character_id"] = new_codename
        choices = load_characters_from_char_dir(state)
        return state, new_codename, f"已重命名：{old_codename} → {new_codename}", gr.update(choices=choices, value=new_codename)

    def handle_batch_save_characters(self, state, char_display):
        if not char_display or not char_display.strip():
            return state, char_display, "没有可保存的角色。", gr.update()
        parts = [part.strip() for part in re.split(r"\n?===\n?", char_display.strip()) if part.strip()]
        if len(parts) <= 1:
            return state, char_display, "未检测到多个角色（用 === 分隔）。请使用「保存角色」保存单个角色。", gr.update()
        story_dir = current_story_dir(state)
        ensure_char_dir(story_dir)
        saved = []
        for part in parts:
            codename = self.next_char_name(state)
            path = character_path(story_dir, codename)
            write_file(path, part)
            state["characters"][codename] = part
            saved.append(codename)
        state["_editing_character_id"] = ""
        state["_pending_character_id"] = ""
        choices = load_characters_from_char_dir(state)
        last = saved[-1] if saved else None
        return state, char_display, f"已批量保存 {len(saved)} 个角色：{', '.join(saved)}", gr.update(choices=choices, value=last)


build_handlers = BuildHandlers()


def handle_world_generate_or_revise(state, concept, world_display):
    return build_handlers.handle_world_generate_or_revise(state, concept, world_display)


def handle_save_world(state, world_display):
    return build_handlers.handle_save_world(state, world_display)


def handle_generate_character(state, char_desc, char_display, char_name):
    return build_handlers.handle_generate_character(state, char_desc, char_display, char_name)


def next_char_name(state):
    return build_handlers.next_char_name(state)


def handle_load_character(state, selected, char_name):
    return build_handlers.handle_load_character(state, selected, char_name)


def handle_save_character(state, char_name, char_display):
    return build_handlers.handle_save_character(state, char_name, char_display)


def handle_delete_character(state, selected):
    return build_handlers.handle_delete_character(state, selected)


def handle_rename_character(state, new_name):
    return build_handlers.handle_rename_character(state, new_name)


def handle_batch_save_characters(state, char_display):
    return build_handlers.handle_batch_save_characters(state, char_display)

