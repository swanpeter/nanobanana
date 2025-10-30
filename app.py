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

DEFAULT_USERNAME = get_secret_value("USERNAME")
DEFAULT_PASSWORD = get_secret_value("PASSWORD")
DEFAULT_GEMINI_API_KEY = get_secret_value("GEMINI_API_KEY") or ""


def get_current_username() -> Optional[str]:
    username = st.session_state.get("config_username")
    if isinstance(username, str) and username.strip():
        return username.strip()
    return DEFAULT_USERNAME


def get_current_password() -> Optional[str]:
    password = st.session_state.get("config_password")
    if isinstance(password, str) and password.strip():
        return password.strip()
    return DEFAULT_PASSWORD


def get_current_api_key() -> Optional[str]:
    api_key = st.session_state.get("config_api_key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return DEFAULT_GEMINI_API_KEY


def render_configuration_controls() -> None:
    with st.expander("設定", expanded=False):
        st.caption(
            "このセッションで利用する Basic 認証情報と Gemini API key を設定できます。"
            "空欄の場合は未設定として扱われます。"
        )

        prev_username = get_current_username()
        prev_password = get_current_password()
        prev_api_key = get_current_api_key()

        with st.form("config_form"):
            username = st.text_input(
                "Basic 認証 ID",
                value=prev_username or "",
                key="config_form_username",
            )
            password = st.text_input(
                "Basic 認証 パスワード",
                value=prev_password or "",
                type="password",
                key="config_form_password",
            )
            api_key = st.text_input(
                "Gemini API key",
                value=prev_api_key or "",
                type="password",
                key="config_form_api_key",
            )
            submitted = st.form_submit_button("設定を保存")
            if submitted:
                normalized_username = username.strip() or None
                normalized_password = password.strip() or None
                normalized_api_key = api_key.strip() or None

                st.session_state["config_username"] = normalized_username
                st.session_state["config_password"] = normalized_password
                st.session_state["config_api_key"] = normalized_api_key

                if (
                    normalized_username != prev_username
                    or normalized_password != prev_password
                ):
                    st.session_state["logged_in"] = False

                st.success("設定を保存しました。")
                rerun_app()


def on_login(expected_username: Optional[str], expected_password: Optional[str]) -> None:
    if (
        st.session_state.get("username_input") == (expected_username or "")
        and st.session_state.get("password_input") == (expected_password or "")
    ):
        st.session_state["logged_in"] = True
        st.success("ログインに成功しました！")
        rerun_app()
    else:
        st.error("ユーザー名またはパスワードが間違っています。")


def login(expected_username: Optional[str], expected_password: Optional[str]) -> None:
    with st.form("login_form"):
        st.text_input("ユーザー名", key="username_input")
        st.text_input("パスワード", type="password", key="password_input")
        submitted = st.form_submit_button("ログイン")
        if submitted:
            on_login(expected_username, expected_password)


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
    st.set_page_config(page_title=TITLE, page_icon="🖼️", layout="centered")
    init_history()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    current_username = get_current_username()
    current_password = get_current_password()

    if not st.session_state["logged_in"]:
        if current_username and current_password:
            st.title("🔑 ログインページ")
            render_configuration_controls()
            login(current_username, current_password)
            return
        st.warning("Basic認証のID/PASSが設定されていないため、ログインなしで利用できます。")
        st.session_state["logged_in"] = True

    st.title("脳内大喜利")
    render_configuration_controls()

    api_key = load_configured_api_key()

    prompt = st.text_area("Prompt", height=150, placeholder="描いてほしい内容を入力してください")
    enforce_no_text = st.toggle("画像にテキストや文字を含めない", value=False)

    if st.button("Generate", type="primary"):
        if not api_key:
            st.warning("設定から Gemini API key を入力してください。")
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
