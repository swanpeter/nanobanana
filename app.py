import base64
from typing import List, Optional, Sequence, Tuple

import streamlit as st

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

USERNAME = get_secret_value("USERNAME")
PASSWORD = get_secret_value("PASSWORD")
DEFAULT_GEMINI_API_KEY = get_secret_value("GEMINI_API_KEY") or ""


def on_login() -> None:
    if st.session_state.get("username_input") == USERNAME and st.session_state.get(
        "password_input"
    ) == PASSWORD:
        st.session_state.logged_in = True
        st.success("ログインに成功しました！")
        rerun_app()
    else:
        st.error("ユーザー名またはパスワードが間違っています。")


def login() -> None:
    with st.form("login_form"):
        st.text_input("ユーザー名", key="username_input")
        st.text_input("パスワード", type="password", key="password_input")
        submitted = st.form_submit_button("ログイン")
        if submitted:
            on_login()


def load_configured_api_key() -> str:
    return get_secret_value("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY


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
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        for part in extract_parts(candidate):
            inline = getattr(part, "inline_data", None)
            if inline is None and isinstance(part, dict):
                inline = part.get("inline_data")
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if data is None and isinstance(inline, dict):
                data = inline.get("data")
            image_bytes = decode_image_data(data)
            if image_bytes:
                return image_bytes
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
        st.session_state.history: List[Tuple[bytes, List[str], str]] = []


def render_history() -> None:
    if not st.session_state.history:
        return

    st.subheader("履歴")
    for image_bytes, texts, model_name in st.session_state.history:
        st.image(
            image_bytes,
            caption=f"{model_name} で生成",
            use_container_width=True,
        )
        for text in texts:
            st.caption(text)
        st.divider()


def main() -> None:
    init_history()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        if USERNAME is None or PASSWORD is None:
            st.session_state.logged_in = True
        else:
            st.set_page_config(page_title="ログイン", page_icon="🔒", layout="centered")
            st.title("🔑 ログインページ")
            login()
            return

    st.set_page_config(page_title=TITLE, page_icon="🖼️", layout="centered")

    if USERNAME is None or PASSWORD is None:
        st.warning("USERNAME / PASSWORD が設定されていないため、ログイン無しで利用できます。")

    st.title("架空大喜利")
    st.subheader("生成AIで笑いを生み出せ！")

    api_key = load_configured_api_key()

    prompt = st.text_area("Prompt", height=150, placeholder="描いてほしい内容を入力してください")
    enforce_no_text = st.toggle("画像にテキストや文字を含めない", value=False)

    if st.button("Generate", type="primary"):
        if not api_key:
            st.warning("Streamlit の設定画面で Gemini API key を設定してください。")
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

        with st.spinner("Gemini が画像を生成しています..."):
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

        texts = collect_text_parts(response)

        st.session_state.history.insert(0, (image_bytes, texts, MODEL_NAME))
        st.success("画像を生成しました。")

    render_history()


if __name__ == "__main__":
    main()
