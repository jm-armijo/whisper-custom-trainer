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
WAVEFORM = STATIC / "waveform.js"
ANALYSIS = STATIC / "audio_analysis.js"

MODULES = (APP, API, STATE, RENDER, WAVEFORM, ANALYSIS, STATIC / "microphone.js")


def code(path):
    """A module's source with its `//` comment lines dropped.

    Every counting assertion reads this rather than the raw text: these files
    explain their constraints in prose that names the very APIs the tests
    forbid, so a grep over the comments passes a test about calling them."""
    return "\n".join(
        line
        for line in path.read_text(encoding="utf8").splitlines()
        if not line.lstrip().startswith("//")
    )

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
        constructions = re.findall(r"new Audio\(", code(APP))
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


class TestTheWaveformNeverCostsATake:
    """The strip is a nicety; a recorded take is the work. Every path into Web
    Audio or a canvas has to be survivable, because a phone that refuses either
    must still record and play."""

    def test_the_analyser_is_torn_down_whenever_recording_stops(self):
        """A rAF loop left running wakes the phone's GPU sixty times a second
        for a canvas nobody is watching, and an AudioContext per take walks
        into the browser's per-page limit."""
        source = code(APP)
        assert "cancelAnimationFrame" in source, "the frame loop is never cancelled"
        assert "analyser.close()" in source, "the AudioContext is never released"
        assert "stopWaveform()" in source

    def test_stopping_a_recording_tears_the_waveform_down(self):
        """The teardown must be reached from stopRecording itself, not only
        from a path an error can skip."""
        source = code(APP)
        body = source[source.index("async function stopRecording()"):]
        assert "stopWaveform()" in body[: body.index("\n}")]

    def test_the_frame_loop_is_cancelled_before_the_context_is_closed(self):
        """A frame already queued would otherwise run against an analyser whose
        context has gone."""
        source = code(APP)
        body = source[source.index("function stopWaveform()"):]
        assert body.index("cancelAnimationFrame") < body.index("analyser.close()")

    def test_a_browser_without_web_audio_still_records(self):
        """isSupported gates the playback draw, and the analyser reports a
        refusal rather than throwing into the record tap."""
        assert "isSupported" in ANALYSIS.read_text(encoding="utf8")
        assert "analysis.isSupported()" in code(APP)

    def test_the_analyser_is_never_wired_to_the_speakers(self):
        """Connecting the mic's analyser to destination feeds the phone's own
        microphone back out of its speaker."""
        source = code(ANALYSIS)
        assert "destination" not in source


class TestPlaybackIsNotRoutedThroughWebAudio:
    """createMediaElementSource silences the element unless the graph is also
    connected to destination, and on iOS that is the fastest way to lose the
    playback the single reused element exists to protect. The playhead is
    driven from the element's own clock instead."""

    def test_create_media_element_source_appears_nowhere(self):
        for module in MODULES:
            assert "createMediaElementSource" not in code(module), module

    def test_the_playhead_follows_the_elements_own_clock(self):
        source = code(APP)
        assert '"timeupdate"' in source
        assert "player.currentTime" in source and "player.duration" in source


class TestTheWaveformIsComputedNotStored:
    """No cache, no sidecar, no new column: dataset.csv keeps exactly the three
    columns train.py reads, and stored peaks would be one more artifact to
    invalidate on every re-record."""

    def test_the_playback_waveform_is_derived_from_the_fetched_clip(self):
        source = code(APP)
        assert "api.fetchAudio(" in source, "the clip is never fetched"
        assert "decodeChannel(" in source, "the fetched bytes are never decoded"
        assert "peaksFromSamples(" in source, "the peaks are never computed"

    def test_the_clip_is_fetched_over_the_documented_audio_route(self):
        """Reusing audioUrl rather than assembling a path by hand keeps the
        cache-busting version and the per-segment escaping the client already
        does; a hand-built URL sent %2F, which matches no route."""
        source = code(API)
        body = source[source.index("export async function fetchAudio"):]
        assert "audioUrl(name, index, version)" in body

    def test_nothing_persists_a_waveform(self):
        """A peak file or a localStorage cache is the regression this forbids."""
        for module in MODULES:
            source = code(module)
            assert "localStorage" not in source, module
            assert "sessionStorage" not in source, module
            assert "indexedDB" not in source, module

    def test_no_module_but_the_api_client_fetches_the_clip(self):
        """UI/logic separation: the canvas layer draws numbers it is handed."""
        for module in (STATE, RENDER, WAVEFORM, ANALYSIS):
            assert "fetch(" not in code(module), module


class TestTheWaveformKeepsTheUiSplit:
    """render.js owns pixels, waveform.js owns arithmetic. The split is what
    lets the peak reduction be tested without a canvas at all."""

    def test_the_waveform_maths_touches_no_canvas(self):
        source = code(WAVEFORM)
        for forbidden in ("document", "getContext", "canvas", "AudioContext"):
            assert forbidden not in source, forbidden

    def test_only_the_view_draws(self):
        """getContext belongs to render.js alone; the controller and the audio
        adapter must not reach for a drawing surface."""
        for module in (APP, ANALYSIS, WAVEFORM, STATE, API):
            assert "getContext" not in code(module), module
        assert 'getContext("2d")' in code(RENDER)

    def test_the_view_takes_its_colours_from_the_stylesheet(self):
        """Data crosses the boundary, not decisions: the palette stays one
        table in style.css rather than being respelled per canvas call."""
        css = STYLE.read_text(encoding="utf8")
        for token in re.findall(r"--waveform-[a-z]+", code(RENDER)):
            assert f"{token}:" in css, token

    def test_the_canvas_the_view_looks_up_exists_in_the_markup(self):
        assert 'id="waveform"' in INDEX.read_text(encoding="utf8")
