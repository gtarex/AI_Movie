import os
import re
import socket

from config import STORIES_ROOT, STORY_DIR as DEFAULT_STORY_DIR


class StoryStorage:
    def __init__(self, stories_root=STORIES_ROOT, default_story_dir=DEFAULT_STORY_DIR):
        self.stories_root = stories_root
        self.default_story_dir = default_story_dir

    def read_file(self, fp):
        if not os.path.exists(fp):
            return ""
        with open(fp, "r", encoding="utf-8") as f:
            return f.read().strip()

    def read_file_raw(self, fp):
        if not os.path.exists(fp):
            return ""
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, fp, content):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content.strip())

    def write_file_raw(self, fp, content):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)

    def append_file(self, fp, content):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "a", encoding="utf-8") as f:
            f.write("\n\n" + content.strip())

    def outline_path(self, story_dir):
        return os.path.join(story_dir, "outline.txt")

    def normalize_story_dir(self, story_dir):
        raw = (story_dir or self.default_story_dir).strip()
        if not raw:
            return self.default_story_dir
        norm = os.path.normpath(raw)
        if os.path.isabs(norm):
            return norm
        root = os.path.normpath(self.stories_root)
        if norm in (".", root):
            return self.default_story_dir
        if norm.startswith(root + os.sep):
            return norm
        if os.sep not in norm:
            return os.path.join(root, norm)
        return norm

    def current_story_dir(self, state):
        if not isinstance(state, dict):
            return self.default_story_dir
        story_dir = self.normalize_story_dir(state.get("story_dir", self.default_story_dir))
        state["story_dir"] = story_dir
        return story_dir

    def list_dirs(self, base=None):
        root = os.path.normpath(base or self.stories_root)
        if root in (".", ""):
            root = os.path.normpath(self.stories_root)
        if not os.path.isdir(root):
            return [self.default_story_dir]
        dirs = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
        ]
        return sorted(dirs) if dirs else [self.default_story_dir]

    def get_char_dir(self, story_dir):
        return os.path.join(story_dir, "char")

    def ensure_char_dir(self, story_dir):
        char_dir = self.get_char_dir(story_dir)
        os.makedirs(char_dir, exist_ok=True)
        return char_dir

    def character_path(self, story_dir, codename):
        return os.path.join(self.get_char_dir(story_dir), f"{codename}.txt")

    def get_current_scene_file(self, story_dir):
        for number in range(1, 1000):
            filename = f"scene_{number:02d}.txt"
            path = os.path.join(story_dir, filename)
            if not os.path.exists(path):
                return path, filename, number
        return os.path.join(story_dir, "scene_999.txt"), "scene_999.txt", 999

    def scene_number_from_name(self, scene_name):
        match = re.match(r"scene_(\d+)\.txt$", scene_name or "")
        return int(match.group(1)) if match else 1

    def list_scene_files(self, story_dir):
        if not os.path.isdir(story_dir):
            return []
        scenes = []
        for filename in os.listdir(story_dir):
            if re.match(r"scene_\d+\.txt$", filename or ""):
                scenes.append(filename)
        return sorted(scenes, key=lambda name: self.scene_number_from_name(name))

    def find_free_port(self, start_port=7860, tries=50):
        for port in range(start_port, start_port + tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        return start_port


storage = StoryStorage()


def read_file(fp):
    return storage.read_file(fp)


def read_file_raw(fp):
    return storage.read_file_raw(fp)


def write_file(fp, content):
    return storage.write_file(fp, content)


def write_file_raw(fp, content):
    return storage.write_file_raw(fp, content)


def append_file(fp, content):
    return storage.append_file(fp, content)


def outline_path(story_dir):
    return storage.outline_path(story_dir)


def normalize_story_dir(story_dir):
    return storage.normalize_story_dir(story_dir)


def current_story_dir(state):
    return storage.current_story_dir(state)


def list_dirs(base=None):
    return storage.list_dirs(base)


def get_char_dir(story_dir):
    return storage.get_char_dir(story_dir)


def ensure_char_dir(story_dir):
    return storage.ensure_char_dir(story_dir)


def character_path(story_dir, codename):
    return storage.character_path(story_dir, codename)


def get_current_scene_file(story_dir):
    return storage.get_current_scene_file(story_dir)


def scene_number_from_name(scene_name):
    return storage.scene_number_from_name(scene_name)


def list_scene_files(story_dir):
    return storage.list_scene_files(story_dir)


def find_free_port(start_port=7860, tries=50):
    return storage.find_free_port(start_port, tries)

