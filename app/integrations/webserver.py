import aioice.ice
_orig_create = aioice.ice.Connection.gather_candidates


async def _patched_gather(self):
    import traceback
    try:
        await _orig_create(self)
    except Exception as e:
        print("gather_candidates EXCEPTION:", e)
        traceback.print_exc()
    print(f"candidates after gather: {len(self.local_candidates)}")

aioice.ice.Connection.gather_candidates = _patched_gather

import asyncio
import fractions
import logging

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

logger = logging.getLogger(__name__)

relay = MediaRelay()
track = None  # set by create_track() before the server starts
peers = set()


class FrameTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, fps):
        super().__init__()
        self.fps = fps
        self.pts = 0
        self.time_base = fractions.Fraction(1, 90_000)
        self.new_frame = asyncio.Event()
        self.latest = None

    def push(self, bgr):
        self.latest = bgr.copy()  # copy so the main loop can't mutate it mid-send
        try:
            asyncio.get_event_loop().call_soon_threadsafe(self.new_frame.set)
        except RuntimeError:
            pass

    async def recv(self):
        # Don't proceed until we actually have a frame
        while self.latest is None:
            await asyncio.sleep(0.01)

        await self.new_frame.wait()
        self.new_frame.clear()

        frame = VideoFrame.from_ndarray(self.latest[..., ::-1], format="rgb24")
        frame.pts = self.pts
        frame.time_base = self.time_base
        self.pts += int(90_000 / self.fps)
        return frame  # no sleep here


track = None


def create_track(fps):
    # just store fps, actual track created inside the loop
    global _fps
    _fps = fps


_fps = 25


def start_web_server(port):
    global track
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Create track HERE, inside the loop that will run it
    track = FrameTrack(_fps)
    print("Track created, loop id:", id(loop))

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    async def run():
        print("Running in loop id:", id(asyncio.get_event_loop()))
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        logger.info("Streaming on http://0.0.0.0:%d", port)
        await asyncio.Event().wait()

    loop.run_until_complete(run())


async def offer(request):
    print("Offer handler loop id:", id(asyncio.get_event_loop()))
    params = await request.json()
    sdp = params["sdp"]
    sdp = "\r\n".join(
        line for line in params["sdp"].splitlines()
        if line != "a=ice-options:trickle"
    ) + "\r\n"
    print("trickle in raw sdp:", "trickle" in sdp)
    print("repr of trickle line:", repr([l for l in sdp.splitlines() if "trickle" in l]))

    # aiortc doesn't support trickle ICE — strip it so aiortc
    # embeds all candidates directly in the answer SDP
    sdp = params["sdp"].replace("a=ice-options:trickle\r\n", "")

    pc = RTCPeerConnection()
    print("PC created")
    peers.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        print("connection state:", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            peers.discard(pc)

    pc.addTrack(relay.subscribe(track))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    print("setLocalDescription done, iceGatheringState:", pc.iceGatheringState)

    print("Answer SDP candidates:")
    for line in pc.localDescription.sdp.splitlines():
        if line.startswith("a=candidate") or line.startswith("a=end-of-candidates"):
            print(" ", line)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def index(_):
    return web.Response(text=HTML, content_type="text/html")


async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in peers])


HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Live Stream</title>
  <style>
    body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }
    video { max-width: 100%; max-height: 100vh; }
  </style>
</head>
<body>
  <video id="v" autoplay muted playsinline></video>
  <script>
    const pc = new RTCPeerConnection({
      iceServers: [],  // no STUN, force local candidates only
      iceTransportPolicy: "all"
    });
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.ontrack = e => document.getElementById("v").srcObject = e.streams[0];

    (async () => {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await new Promise(r => { if (pc.iceGatheringState === "complete") r(); else pc.onicegatheringstatechange = () => pc.iceGatheringState === "complete" && r(); });
      const res = await fetch("/offer", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }) });
      await pc.setRemoteDescription(await res.json());
    })();
  </script>
</body>
</html>"""
