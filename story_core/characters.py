import os
import re

from .storage import ensure_char_dir


CHARACTER_FIELDS = ["性别", "年龄", "职业", "外貌", "性格特征", "爱好", "过往经历", "人际关系", "能力"]


class CharacterProfileService:
    def sanitize_codename(self, raw_name):
        codename = "".join(c if c.isalnum() else "_" for c in raw_name.strip().lower())
        codename = re.sub(r"_+", "_", codename).strip("_")
        return codename or "unknown_character"

    def unique_character_codename(self, story_dir, codename):
        char_dir = ensure_char_dir(story_dir)
        base = self.sanitize_codename(codename)
        candidate = base
        index = 1
        while os.path.exists(os.path.join(char_dir, f"{candidate}.txt")):
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def extract_character_id(self, profile):
        match = re.search(r"姓名\s*[:：]\s*.*?[（(]\s*([A-Za-z0-9_]+)\s*[）)]", profile or "")
        return self.sanitize_codename(match.group(1)) if match else ""

    def extract_character_name(self, profile):
        match = re.search(r"姓名\s*[:：]\s*([^（(\n]+)", profile or "")
        return match.group(1).strip() if match else ""

    def local_codename_from_text(self, text, fallback="new_character"):
        text = (text or "").strip()
        ascii_part = "".join(c.lower() if c.isalnum() else "_" for c in text if c.isascii())
        ascii_part = re.sub(r"_+", "_", ascii_part).strip("_")
        if ascii_part:
            return self.sanitize_codename(ascii_part)
        return self.sanitize_codename(fallback)

    def normalize_character_profile(self, profile, codename, fallback_name="未命名角色"):
        codename = self.sanitize_codename(codename)
        values = {field: "" for field in CHARACTER_FIELDS}
        name = self.extract_character_name(profile) or fallback_name
        current_field = None
        field_pattern = re.compile(rf"^({'|'.join(CHARACTER_FIELDS)})\s*[:：]?\s*(.*)$")

        for raw_line in (profile or "").replace("\r\n", "\n").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("姓名"):
                parsed_name = self.extract_character_name(line)
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


character_profiles = CharacterProfileService()


def sanitize_codename(raw_name):
    return character_profiles.sanitize_codename(raw_name)


def unique_character_codename(story_dir, codename):
    return character_profiles.unique_character_codename(story_dir, codename)


def extract_character_id(profile):
    return character_profiles.extract_character_id(profile)


def extract_character_name(profile):
    return character_profiles.extract_character_name(profile)


def local_codename_from_text(text, fallback="new_character"):
    return character_profiles.local_codename_from_text(text, fallback)


def normalize_character_profile(profile, codename, fallback_name="未命名角色"):
    return character_profiles.normalize_character_profile(profile, codename, fallback_name)

