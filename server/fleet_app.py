from openenv.core.env_server import create_fastapi_app
from .fleet_environment import FleetResumeEnvironment
from models import FleetAction, FleetObservation

# Create the FastAPI app for the multi-agent fleet environment
app = create_fastapi_app(
    FleetResumeEnvironment,
    action_cls=FleetAction,
    observation_cls=FleetObservation,
)

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
import logging

logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def home():
    """Home page for the Hugging Face Space."""
    return """
    <html>
        <head>
            <title>Hiring Fleet — AI Oversight System</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding: 50px; background: #0f172a; color: #e2e8f0; }
                .card { background: #1e293b; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); display: inline-block; max-width: 560px; }
                h1 { color: #f8fafc; margin-bottom: 4px; }
                .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 20px; }
                .status { color: #34d399; font-weight: bold; }
                .link { color: #60a5fa; text-decoration: none; }
                .badge { display: inline-block; background: #7c3aed; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; margin: 4px; }
                code { background: #0f172a; padding: 10px; display: block; border-radius: 6px; text-align: left; font-size: 0.85rem; line-height: 1.7; }
                hr { border-color: #334155; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🤖 Hiring Fleet</h1>
                <p class="subtitle">AI Oversight System — Multi-Agent Resume Screening</p>
                <p>Status: <span class="status">ONLINE</span></p>
                <p style="font-size: 0.8rem; color: #64748b;">Version: v3.0.0</p>
                <div>
                    <span class="badge">Fraud Specialist</span>
                    <span class="badge">Skills Specialist</span>
                    <span class="badge">Timeline Specialist</span>
                    <span class="badge">Overseer</span>
                </div>
                <hr>
                <p>Endpoints:</p>
                <code>
                    POST /reset<br>
                    POST /step<br>
                    GET  /health
                </code>
                <p style="margin-top: 20px;"><a class="link" href="/docs">View API Documentation</a></p>
            </div>
        </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Entry point for the fleet environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
