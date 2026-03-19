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


def create_track(fps):
    global track
    track = FrameTrack(fps)
    return track


async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    peers.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        print("connection state:", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            peers.discard(pc)

    pc.addTrack(relay.subscribe(track))
    await pc.setRemoteDescription(RTCSessionDescription(**params))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Wait for ICE gathering to complete so all candidates
    # are included in the answer SDP we return to the browser
    ice_done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    async def on_ice():
        if pc.iceGatheringState == "complete":
            ice_done.set()

    if pc.iceGatheringState != "complete":
        await asyncio.wait_for(ice_done.wait(), timeout=10)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def index(_):
    return web.Response(text=HTML, content_type="text/html")


async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in peers])


def start_web_server(port):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    async def run():
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        logger.info("Streaming on http://0.0.0.0:%d", port)
        await asyncio.Event().wait()

    loop.run_until_complete(run())


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
