import base64
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except ImportError:
    StreamlitSecretNotFoundError = Exception

try:
    from google import genai
    from google.api_core import exceptions as google_exceptions
    from google.genai import types
except ImportError:
    st.error(
        "必要なライブラリが不足しています。`pip install -r requirements.txt` を実行してください。"
    )
    st.stop()

def get_secret_value(key: str) -> Optional[str]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None
    except Exception:
        return None
    try:
        return secrets_obj[key]
    except (KeyError, TypeError, StreamlitSecretNotFoundError):
        pass
    get_method = getattr(secrets_obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key)
        except Exception:
            return None
    return None


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


TITLE = "Gemini 画像生成"
MODEL_NAME = "models/gemini-2.5-flash-image-preview"
DEFAULT_PROMPT_SUFFIX = (
    "((masterpiece, best quality, ultra-detailed, photorealistic, 8k, sharp focus))"
)
NO_TEXT_TOGGLE_SUFFIX = (
    "((no background text, no symbols, no markings, no letters anywhere, no typography, "
    "no signboard, no watermark, no logo, no text, no subtitles, no labels, no poster elements, neutral background))"
)

DEFAULT_GEMINI_API_KEY = (
    get_secret_value("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
)


def _normalize_credential(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def get_secret_auth_credentials() -> Tuple[Optional[str], Optional[str]]:
    try:
        secrets_obj = st.secrets
    except StreamlitSecretNotFoundError:
        return None, None
    except Exception:
        return None, None

    auth_section: Optional[Dict[str, Any]] = None
    if isinstance(secrets_obj, dict):
        auth_section = secrets_obj.get("auth")
    else:
        auth_section = getattr(secrets_obj, "get", lambda _key, _default=None: None)("auth")

    def _get_from_container(container: object, key: str) -> Optional[Any]:
        if isinstance(container, dict):
            return container.get(key)
        getter = getattr(container, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                try:
                    return getter(key, None)
                except TypeError:
                    return None
        try:
            return getattr(container, key)
        except AttributeError:
            return None

    def _extract_credential(container: object, keys: Tuple[str, ...]) -> Optional[Any]:
        for key in keys:
            value = _get_from_container(container, key)
            if value is not None:
                return value
        return None

    username = None
    password = None
    if auth_section is not None:
        username = _extract_credential(auth_section, ("username", "id", "user", "name"))
        password = _extract_credential(auth_section, ("password", "pass", "pwd"))

    if username is None:
        username = get_secret_value("USERNAME") or get_secret_value("ID")
    if password is None:
        password = get_secret_value("PASSWORD") or get_secret_value("PASS")

    normalized_username = _normalize_credential(str(username)) if username is not None else None
    normalized_password = _normalize_credential(str(password)) if password is not None else None
    return normalized_username, normalized_password


def get_configured_auth_credentials() -> Tuple[str, str]:
    secret_username, secret_password = get_secret_auth_credentials()
    if secret_username and secret_password:
        return secret_username, secret_password
    return "mezamashi", "mezamashi"


def require_login() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return

    st.title("ログイン")

    username, password = get_configured_auth_credentials()
    if not username or not password:
        st.info("ログイン情報が未設定です。管理者に連絡してください。")
        st.stop()
        return

    with st.form("login_form", clear_on_submit=False):
        input_username = st.text_input("ID")
        input_password = st.text_input("PASS", type="password")
        submitted = st.form_submit_button("ログイン")

    if submitted:
        if input_username == username and input_password == password:
            st.session_state["authenticated"] = True
            st.success("ログインしました。")
            rerun_app()
            return
        st.error("IDまたはPASSが正しくありません。")
    st.stop()


def get_current_api_key() -> Optional[str]:
    api_key = st.session_state.get("config_api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return DEFAULT_GEMINI_API_KEY


def load_configured_api_key() -> str:
    return get_current_api_key() or ""


def decode_image_data(data: Optional[object]) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except (ValueError, TypeError):
            return None
    return None


def extract_parts(candidate: object) -> Sequence:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if parts is None and isinstance(candidate, dict):
        parts = candidate.get("content", {}).get("parts", [])
    return parts or []


def collect_image_bytes(response: object) -> Optional[bytes]:
    visited: set[int] = set()
    queue: List[object] = []

    if response is not None:
        queue.append(response)

    def handle_inline(container: object) -> Optional[bytes]:
        if container is None:
            return None
        data = getattr(container, "data", None)
        if data is None and isinstance(container, dict):
            data = container.get("data")
        return decode_image_data(data)

    def maybe_file_data(container: object) -> Optional[bytes]:
        if container is None:
            return None
        file_data = getattr(container, "file_data", None)
        if file_data is None and isinstance(container, dict):
            file_data = container.get("file_data")
        if file_data:
            data = getattr(file_data, "data", None)
            if data is None and isinstance(file_data, dict):
                data = file_data.get("data")
            decoded = decode_image_data(data)
            if decoded:
                return decoded
        return None

    base64_charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")

    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        if isinstance(current, bytes):
            if current:
                return current
            continue

        if isinstance(current, (bytearray, memoryview)):
            as_bytes = bytes(current)
            if as_bytes:
                return as_bytes
            continue

        if isinstance(current, str):
            candidate = current.strip()
            if len(candidate) > 80 and set(candidate) <= base64_charset:
                decoded = decode_image_data(candidate)
                if decoded:
                    return decoded
            continue

        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        if isinstance(current, dict):
            inline = current.get("inline_data")
            decoded = handle_inline(inline)
            if decoded:
                return decoded

            decoded = maybe_file_data(current)
            if decoded:
                return decoded

            for key, value in current.items():
                if key in {"data", "image", "blob"}:
                    decoded = decode_image_data(value)
                    if decoded:
                        return decoded
                queue.append(value)
            continue

        decoded = handle_inline(getattr(current, "inline_data", None))
        if decoded:
            return decoded

        decoded = maybe_file_data(current)
        if decoded:
            return decoded

        for attr in (
            "candidates",
            "content",
            "parts",
            "generated_content",
            "contents",
            "responses",
            "messages",
            "media",
            "image",
            "images",
        ):
            value = getattr(current, attr, None)
            if value is not None:
                queue.append(value)

        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray, memoryview)):
            queue.extend(list(current))

    return None


def collect_text_parts(response: object) -> List[str]:
    texts: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        for part in extract_parts(candidate):
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                texts.append(text)
    return texts


def init_history() -> None:
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, object]] = []


def ensure_lightbox_assets() -> None:
    components.html(
        """
        <script>
        (function () {
            const parentWindow = window.parent;
            if (!parentWindow) {
                return;
            }

            try {
                delete parentWindow.__streamlitLightbox;
            } catch (err) {
                parentWindow.__streamlitLightbox = undefined;
            }
            parentWindow.__streamlitLightboxInitialized = false;
            const doc = parentWindow.document;

            if (!doc.getElementById("streamlit-lightbox-style")) {
                const style = doc.createElement("style");
                style.id = "streamlit-lightbox-style";
                style.textContent = `
                .streamlit-lightbox-thumb {
                    width: 100%;
                    display: block;
                    border-radius: 12px;
                    cursor: pointer;
                    transition: transform 0.16s ease-in-out;
                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
                    margin: 0 auto 0.75rem auto;
                }
                .streamlit-lightbox-thumb:hover {
                    transform: scale(1.02);
                }
                `;
                doc.head.appendChild(style);
            }

            parentWindow.__streamlitLightbox = (function () {
                let overlay = null;
                let keyHandler = null;

                function hide() {
                    if (!overlay) {
                        return;
                    }
                    overlay.style.opacity = "0";
                    const originalOverflow = overlay.getAttribute("data-original-overflow") || "";
                    doc.body.style.overflow = originalOverflow;
                    setTimeout(function () {
                        if (overlay && overlay.parentNode) {
                            overlay.parentNode.removeChild(overlay);
                        }
                        overlay = null;
                    }, 180);
                    if (keyHandler) {
                        parentWindow.removeEventListener("keydown", keyHandler);
                        keyHandler = null;
                    }
                }

                function show(src) {
                    hide();
                    overlay = doc.createElement("div");
                    overlay.id = "streamlit-lightbox-overlay";
                    overlay.style.position = "fixed";
                    overlay.style.zIndex = "10000";
                    overlay.style.top = "0";
                    overlay.style.left = "0";
                    overlay.style.right = "0";
                    overlay.style.bottom = "0";
                    overlay.style.display = "flex";
                    overlay.style.justifyContent = "center";
                    overlay.style.alignItems = "center";
                    overlay.style.background = "rgba(0, 0, 0, 0.92)";
                    overlay.style.cursor = "zoom-out";
                    overlay.style.opacity = "0";
                    overlay.style.transition = "opacity 0.18s ease-in-out";
                    overlay.setAttribute("data-original-overflow", doc.body.style.overflow || "");
                    doc.body.style.overflow = "hidden";

                    const full = doc.createElement("img");
                    full.src = src;
                    full.alt = "Generated image fullscreen";
                    full.style.maxWidth = "100vw";
                    full.style.maxHeight = "100vh";
                    full.style.objectFit = "contain";
                    full.style.boxShadow = "0 20px 45px rgba(0, 0, 0, 0.5)";
                    full.style.borderRadius = "0";

                    overlay.appendChild(full);
                    overlay.addEventListener("click", hide);

                    keyHandler = function (event) {
                        if (event.key === "Escape") {
                            hide();
                        }
                    };
                    parentWindow.addEventListener("keydown", keyHandler);

                    doc.body.appendChild(overlay);
                    requestAnimationFrame(function () {
                        overlay.style.opacity = "1";
                    });
                }

                return { show, hide };
            })();
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def render_clickable_image(image_bytes: bytes, element_id: str) -> None:
    ensure_lightbox_assets()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    image_src = f"data:image/png;base64,{encoded}"
    image_src_json = json.dumps(image_src)
    components.html(
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        img {{
            width: 100%;
            display: block;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.16s ease-in-out;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        }}
        img:hover {{
            transform: scale(1.02);
        }}
    </style>
</head>
<body>
    <img id="thumb" src="{image_src}" alt="Generated image">
    <script>
    (function() {{
        const img = document.getElementById("thumb");
        if (!img) {{
            return;
        }}

        function resizeFrame() {{
            const frame = window.frameElement;
            if (!frame) {{
                return;
            }}
            const frameWidth = frame.getBoundingClientRect().width || img.naturalWidth || img.clientWidth || 0;
            const ratio = img.naturalWidth ? (img.naturalHeight / Math.max(img.naturalWidth, 1)) : (img.clientHeight / Math.max(img.clientWidth, 1) || 1);
            const height = frameWidth ? Math.max(160, frameWidth * ratio) : (img.clientHeight || img.naturalHeight || 320);
            frame.style.height = height + "px";
        }}

        if (img.complete) {{
            resizeFrame();
        }} else {{
            img.addEventListener("load", resizeFrame);
        }}
        window.addEventListener("resize", resizeFrame);
        setTimeout(resizeFrame, 60);

        img.addEventListener("click", function() {{
            if (window.parent && window.parent.__streamlitLightbox) {{
                window.parent.__streamlitLightbox.show({image_src_json});
            }}
        }});
    }})();
    </script>
</body>
</html>
""",
        height=200,
        scrolling=False,
    )


def render_history() -> None:
    if not st.session_state.history:
        return

    st.subheader("履歴")
    for entry in st.session_state.history:
        image_bytes = entry.get("image_bytes")
        prompt_text = entry.get("prompt") or ""
        if image_bytes:
            image_id = entry.get("id")
            if not isinstance(image_id, str):
                image_id = f"img_{uuid.uuid4().hex}"
                entry["id"] = image_id
            render_clickable_image(image_bytes, image_id)
        prompt_display = prompt_text.strip()
        st.markdown("**Prompt**")
        st.write(prompt_display if prompt_display else "(未入力)")
        st.divider()


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🖼️", layout="centered")
    init_history()
    require_login()

    st.title("脳内大喜利")

    api_key = load_configured_api_key()

    prompt = st.text_area("Prompt", height=150, placeholder="描いてほしい内容を入力してください")
    enforce_no_text = st.toggle("画像にテキストや文字を含めない", value=False)

    if st.button("Generate", type="primary"):
        if not api_key:
            st.warning("Gemini API key が設定されていません。Streamlit secrets などで設定してください。")
            st.stop()
        if not prompt.strip():
            st.warning("プロンプトを入力してください。")
            st.stop()

        client = genai.Client(api_key=api_key.strip())
        stripped_prompt = prompt.rstrip()
        prompt_for_request = (
            f"{stripped_prompt}\n{DEFAULT_PROMPT_SUFFIX}"
            if stripped_prompt
            else DEFAULT_PROMPT_SUFFIX
        )
        if enforce_no_text:
            prompt_for_request = f"{prompt_for_request}\n{NO_TEXT_TOGGLE_SUFFIX}"

        with st.spinner("画像を生成しています..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt_for_request,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                    ),
                )
            except google_exceptions.ResourceExhausted:
                st.error(
                    "Gemini API のクォータ（無料枠または請求プラン）を超えました。"
                    "しばらく待つか、Google AI Studio で利用状況と請求設定を確認してください。"
                )
                st.info("https://ai.google.dev/gemini-api/docs/rate-limits")
                st.stop()
            except google_exceptions.GoogleAPICallError as exc:
                st.error(f"API 呼び出しに失敗しました: {exc.message}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"予期しないエラーが発生しました: {exc}")
                st.stop()

        image_bytes = collect_image_bytes(response)
        if not image_bytes:
            st.error("画像データを取得できませんでした。")
            st.stop()

        user_prompt = prompt.strip()
        st.session_state.history.insert(
            0,
            {
                "id": f"img_{uuid.uuid4().hex}",
                "image_bytes": image_bytes,
                "prompt": user_prompt,
                "model": MODEL_NAME,
                "no_text": enforce_no_text,
            },
        )
        st.success("画像を生成しました。")

    render_history()


if __name__ == "__main__":
    main()
