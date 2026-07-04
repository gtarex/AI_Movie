import html
import json
import re


class TextTools:
    def extract_json_object(self, raw_response):
        if not raw_response:
            return {}
        raw_response = raw_response.strip()
        raw_response = re.sub(r"^```(?:json)?\s*\n?", "", raw_response)
        raw_response = re.sub(r"\n?```\s*$", "", raw_response).strip()
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        try:
            return json.loads(match.group(0) if match else raw_response)
        except json.JSONDecodeError:
            return {}

    def limit_words(self, text, max_words=200):
        words = text.strip().split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).strip()

    def limit_chars(self, text, max_chars=200):
        text = text.strip()
        return text if len(text) <= max_chars else text[:max_chars].strip()


text_tools = TextTools()


def extract_json_object(raw_response):
    return text_tools.extract_json_object(raw_response)


def limit_words(text, max_words=200):
    return text_tools.limit_words(text, max_words)


def limit_chars(text, max_chars=200):
    return text_tools.limit_chars(text, max_chars)


def story_message_box(content, status="draft", title=""):
    content = str(content or "").strip()
    if not content:
        return ""
    if status == "accepted":
        cls = "story-ai-accepted"
        default_title = "已通过"
    else:
        cls = "story-ai-draft"
        default_title = "未通过草稿"
    safe_title = html.escape(title or default_title)
    safe_content = html.escape(content)
    return (
        f'<div class="{cls}">'
        f'<div class="story-ai-status">{safe_title}</div>'
        f'<div class="story-ai-body">{safe_content}</div>'
        f'</div>'
    )
