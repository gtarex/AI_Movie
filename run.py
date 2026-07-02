#!/usr/bin/env python3
"""Launch the DeepSeek interactive novel UI."""

from webui import build_ui, find_free_port


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=find_free_port(), inbrowser=True, share=False)
