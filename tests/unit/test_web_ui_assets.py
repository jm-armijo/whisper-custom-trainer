"""Static checks on the browser recorder's assets.

A browser cannot be driven from the test suite, so these prove only what is
checkable without one: the assets exist, the API client covers the whole
server contract, the four chunk statuses all have a colour, and nothing loads
from a CDN. Real rendering is verified by opening the page - the same
limitation the curses view has with its stub screen.

Deliberately absent: any assertion about the *spelling* of an asset URL. That
kind of check only restates what the file says, and it pinned a /static/
prefix the server did not serve while every script on the page 404'd.
Whether an asset URL resolves is proved against a live server in
tests/integration/test_recorder_server_http.py::TestTheRealPageLoads.
"""

import re

import whisper_pipeline as wp

STATIC = wp.PROJECT_ROOT / "static"

INDEX = STATIC / "index.html"
APP = STATIC / "app.js"
STYLE = STATIC / "style.css"
API = STATIC / "api.js"
STATE = STATIC / "state.js"
RENDER = STATIC / "render.js"

MODULES = (APP, API, STATE, RENDER, STATIC / "microphone.js")

# The routes recorder_server.py serves; the client must address every one.
# scriptPath escapes each segment of "<language>/<file>" separately, so the one
# structural slash survives; encodeURIComponent over the whole name sent %2F,
# which matches no route.
CONTRACT_PATHS = (
    "/api/scripts",
    "/api/scripts/${scriptPath(name)}",
    "/api/scripts/${scriptPath(name)}/chunks/${index}",
    "/api/scripts/${scriptPath(name)}/chunks/${index}/audio",
)

# recorder_state.chunk_statuses returns four, not three: a recorded line under
# the cursor must not be painted as "read this next".
STATUSES = ("recorded", "selected", "recorded_selected", "pending")


class TestAssetsExist:
    def test_every_asset_the_page_needs_is_present(self):
        assert INDEX.exists()
        assert STYLE.exists()
        for module in MODULES:
            assert module.exists(), module

    def test_the_entry_script_is_loaded_as_a_module(self):
        """Whether each asset URL resolves is proved by fetching it from a live
        server in tests/integration; only the module type is checkable here,
        and app.js uses import, so a classic script tag would not run at all."""
        assert 'type="module"' in INDEX.read_text(encoding="utf8")

    def test_every_element_id_the_view_reads_exists_in_the_markup(self):
        markup = INDEX.read_text(encoding="utf8")
        wanted = re.findall(r'getElementById\("([^"]+)"\)', RENDER.read_text(encoding="utf8"))
        assert wanted, "render.js should look up its elements by id"
        missing = [name for name in wanted if f'id="{name}"' not in markup]
        assert missing == []


class TestApiContract:
    def test_the_client_addresses_every_documented_route(self):
        source = API.read_text(encoding="utf8")
        for path in CONTRACT_PATHS:
            assert path in source, path

    def test_a_take_is_uploaded_and_deleted_over_the_chunk_route(self):
        source = API.read_text(encoding="utf8")
        assert 'method: "POST"' in source
        assert 'method: "DELETE"' in source

    def test_fetch_lives_only_in_the_api_client(self):
        """UI/logic separation: the view and the domain never reach the network."""
        for module in (STATE, RENDER):
            assert "fetch(" not in module.read_text(encoding="utf8"), module


class TestStatusColours:
    def test_all_four_chunk_statuses_are_styled_distinctly(self):
        css = STYLE.read_text(encoding="utf8")
        for status in STATUSES:
            assert f".chunk--{status}" in css, status

    def test_recorded_and_recorded_selected_are_not_the_same_rule(self):
        css = STYLE.read_text(encoding="utf8")
        assert ".chunk--recorded_selected" in css
        # A shared selector would collapse the distinction chunk_statuses draws.
        assert ".chunk--recorded,.chunk--recorded_selected" not in css.replace(" ", "")

    def test_the_domain_module_names_statuses_and_leaves_colours_to_the_view(self):
        source = STATE.read_text(encoding="utf8")
        for status in STATUSES:
            assert f'"{status}"' in source, status
        # Data crosses the boundary, not decisions: the domain returns
        # "recorded_selected", never a hex colour or a CSS class.
        assert re.search(r"#[0-9a-fA-F]{3,8}\b", source) is None
        assert "chunk--" not in source


class TestNoExternalDependencies:
    def test_no_asset_loads_anything_over_the_network(self):
        """No CDN, no build step: the recorder must work on a phone with the
        laptop's local server as the only reachable host."""
        for path in (INDEX, STYLE, *MODULES):
            text = path.read_text(encoding="utf8")
            assert "http://" not in text, path
            assert "https://" not in text, path


class TestPlaybackSurvivesIosGesturePolicy:
    """iOS ties permission to play to the element that was live during the
    user's tap. A fresh Audio object per tap is unpermitted, so its play()
    promise rejects with NotAllowedError and the phone stays silent - which is
    invisible from the desktop, where every approach works."""

    def test_one_audio_element_is_reused(self):
        # Comment lines are dropped first: this file explains the policy in
        # prose that names the very construction it forbids.
        code = "\n".join(
            line for line in APP.read_text(encoding="utf8").splitlines()
            if not line.lstrip().startswith("//")
        )
        constructions = re.findall(r"new Audio\(", code)
        assert len(constructions) == 1, "a per-tap Audio object loses the gesture"

    def test_the_element_is_constructed_without_a_source(self):
        """The src is assigned per take; constructing with a URL would tie the
        one element to whichever take happened to be first."""
        assert "new Audio()" in APP.read_text(encoding="utf8")

    def test_the_rejection_is_reported_to_the_user(self):
        """A silent failure here is the bug itself: the page must say
        something rather than look like it played."""
        source = APP.read_text(encoding="utf8")
        assert "NotAllowedError" in source

    def test_an_aborted_play_is_not_reported_as_a_failure(self):
        """load() rejects the previous tap's play() with AbortError. With one
        reused element that stale rejection would otherwise announce a failure
        over the newer take that is playing correctly."""
        assert "AbortError" in APP.read_text(encoding="utf8")
