# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os

from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

from expense_agent.app_utils.telemetry import setup_telemetry
from expense_agent.app_utils.typing import Feedback

# Configure standard Python logging for console logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("expense_agent")

setup_telemetry()

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

# Initialize FastAPI application with trigger support
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,  # Enables DevServer so dev-ui works while triggers are still active
    trigger_sources=["pubsub"],  # Accepts Pub/Sub trigger messages
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=False,  # Disable cloud telemetry as requested
)
app.title = "ambient-expense-agent"
app.description = "API for interacting with the Agent ambient-expense-agent"


class PubSubNormalizationMiddleware:
    """Middleware to normalize Pub/Sub subscription paths to their short name.

    Pub/Sub sends subscription paths as fully-qualified names (e.g.
    'projects/my-project/subscriptions/my-sub'). This middleware extracts
    the short name (the last segment) to keep session records and user_ids readable.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope["path"].endswith("/trigger/pubsub")
            and scope["method"] == "POST"
        ):
            # Read the whole request body
            body = b""
            more_body = True
            while more_body:
                message = await receive()
                body += message.get("body", b"")
                more_body = message.get("more_body", False)

            if body:
                try:
                    data = json.loads(body)
                    # Normalize subscription path if present
                    if data.get("subscription"):
                        sub_path = data["subscription"]
                        if "/" in sub_path:
                            # Extract the short name (last segment)
                            data["subscription"] = sub_path.split("/")[-1]

                    # Reconstruct request body
                    body = json.dumps(data).encode("utf-8")
                except Exception as e:
                    logger.warning("Failed to normalize Pub/Sub subscription: %s", e)

            # Define a new receive channel that feeds the modified body
            async def new_receive():
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }

            await self.app(scope, new_receive, send)
        else:
            await self.app(scope, receive, send)


# Add the subscription normalization middleware
app.add_middleware(PubSubNormalizationMiddleware)


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.info("Feedback received: %s", feedback.model_dump())
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
