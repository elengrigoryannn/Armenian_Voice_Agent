import os
import secrets
import string

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from livekit import api

load_dotenv()

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
DEFAULT_ROOM = "bank-demo"
PORT = 8000

app = FastAPI()


def random_identity(prefix="user"):
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{prefix}-{suffix}"


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/token")
def token(room: str = DEFAULT_ROOM, identity: str | None = None):
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    livekit_url = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set. "
                   "Copy .env.example to .env and fill them in.",
        )

    identity = identity or random_identity()

    jwt = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

    return {
        "url": livekit_url,
        "room": room,
        "identity": identity,
        "token": jwt,
    }


if __name__ == "__main__":
    print(f"\nServing web client + token endpoint at http://localhost:{PORT}")
    print("Make sure voice_agent.py is also running (in another terminal) "
          "so an agent joins the room you connect to.\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
