"""
Flask Route Definitions for Browser Service API.

This module contains all Flask route handlers for the browser automation service.
It separates API endpoint definitions from business logic, making the codebase
more maintainable and testable.

Routes:
    GET  /           - Service information and available endpoints
    GET  /health     - Health check with service status
    GET  /probe      - Legacy health check endpoint
    POST /workflow   - Submit workflow task (primary endpoint)
    POST /batch      - Deprecated alias for /workflow
    GET  /query/<id> - Query task status by ID
    GET  /tasks      - List all tasks with summaries
"""

import logging
import time
import uuid
from typing import Any

from flask import Flask, jsonify, request

from browser_service.api.handlers import (
    format_error_response,
    format_task_list_response,
    format_task_response,
    validate_workflow_request,
)
from browser_service.config import config
from browser_service.tasks import process_workflow_task

try:
    from src.backend.core.config import settings as _nl_settings
except ImportError:
    _nl_settings = None

logger = logging.getLogger(__name__)

# The only 5xx text a caller ever sees. Exception detail stays in the log —
# messages carry absolute paths, CDP endpoints and upstream payloads.
INTERNAL_ERROR_MESSAGE = "Internal server error"


def register_routes(app: Flask, task_processor: Any) -> None:
    """
    Register all API routes with the Flask app.

    Args:
        app: Flask application instance
        task_processor: TaskProcessor instance for managing tasks

    Example:
        >>> app = Flask(__name__)
        >>> task_processor = TaskProcessor(executor)
        >>> register_routes(app, task_processor)
    """

    @app.route("/", methods=["GET"])
    def root():
        """Root endpoint to verify service is running."""
        return jsonify(
            {
                "service": "Enhanced Browser Use Service with Vision-Based Locators",
                "status": "running",
                "version": "4.0.0",
                "improvements": [
                    "Vision AI for element identification (built-in browser-use)",
                    "Structured JSON locator output",
                    "Multiple locator strategies (10+ options)",
                    "Locator stability scoring",
                    "Validation and uniqueness checking",
                    "Smart fallback mechanisms",
                    "Better encoding handling",
                    "Proper session cleanup",
                    "NEW: Batch processing mode for multiple elements in one session",
                    "NEW: Persistent browser session across element lookups",
                    "NEW: Context-aware popup handling",
                ],
                "endpoints": [
                    "GET / - This endpoint",
                    "GET /health - Health check",
                    "GET /probe - Legacy health check",
                    "POST /workflow - Process workflow task (unified session, RECOMMENDED)",
                    "POST /batch - Deprecated alias for /workflow (backward compatible)",
                    "GET /query/<task_id> - Query task status",
                    "GET /tasks - List all tasks",
                ],
            }
        ), 200

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint with capacity information."""
        active_count = task_processor.count_active_tasks()
        max_tasks = config.max_concurrent_tasks
        return jsonify(
            {
                "status": "healthy",
                "service": "enhanced_browser_use_service",
                "timestamp": time.time(),
                "active_tasks": active_count,
                "max_tasks": max_tasks,
                "available_slots": max_tasks - active_count,
                "tasks_submitted": task_processor.tasks_submitted_count(),
                "encoding": "utf-8",
                "model_provider": config.llm.model_provider,
                "headless": config.headless,
                "google_api_configured": (
                    bool(
                        config.llm.google_api_key
                        and config.llm.google_api_key != "your_api_key_here"
                    )
                    if config.llm.model_provider == "gemini"
                    else config.llm.vertexai_credentials is not None
                    if config.llm.model_provider == "vertex"
                    else False
                ),
            }
        ), 200

    @app.route("/probe", methods=["GET"])
    def probe():
        """Legacy probe endpoint for backward compatibility."""
        return jsonify({"status": "alive", "message": "enhanced_browser_use_service is alive"}), 200

    @app.route("/workflow", methods=["POST"])
    @app.route("/batch", methods=["POST"])  # Deprecated alias for backward compatibility
    def workflow_submit():
        """
        Process a workflow task with multiple elements in a single browser session.

        This endpoint handles complete user workflows (navigate → act → extract locators).
        All elements are processed in ONE browser session for context preservation.

        Endpoints:
            /workflow - Primary endpoint (recommended)
            /batch - Deprecated alias (backward compatible)

        Request JSON:
            {
                "elements": [{"id": "elem_1", "description": "...", "action": "input"}, ...],
                "url": "https://example.com",
                "user_query": "search for shoes and get product name",
                "enable_custom_actions": true  // Optional: Enable/disable custom actions (defaults to config value)
            }

        Response JSON:
            {
                "task_id": "uuid",
                "status": "processing",
                "message": "Workflow task submitted (N elements in single session)"
            }
        """
        # Log deprecation warning if /batch endpoint is used
        if request.path == "/batch":
            logger.warning("⚠️  /batch endpoint is deprecated. Please use /workflow instead.")

        logger.info(f"📥 Received workflow request via {request.path}")

        try:
            # Validate request
            is_valid, error_message, data = validate_workflow_request(request)
            if not is_valid:
                return format_error_response(error_message, 400)

            # Extract validated data
            elements = data["elements"]
            url = data["url"]
            user_query = data["user_query"]
            session_config = data["session_config"]
            enable_custom_actions = data["enable_custom_actions"]
            parent_workflow_id = data.get("parent_workflow_id")  # Optional
            org_id = data.get("org_id")  # Optional: correlation/observability
            user_id = data.get("user_id")  # Optional: correlation/observability

            # Feature flag: enable_custom_actions (defaults to config value if not provided)
            if enable_custom_actions is None:
                if _nl_settings is not None and hasattr(_nl_settings, "ENABLE_CUSTOM_ACTIONS"):
                    enable_custom_actions = _nl_settings.ENABLE_CUSTOM_ACTIONS
                else:
                    enable_custom_actions = config.enable_custom_actions
                logger.info(
                    f"🔧 enable_custom_actions not provided in request, using config default: {enable_custom_actions}"
                )
            else:
                logger.info(
                    f"🔧 enable_custom_actions provided in request: {enable_custom_actions}"
                )

            # Log parent_workflow_id if provided
            if parent_workflow_id:
                logger.info(
                    f"📎 Parent workflow ID provided: {parent_workflow_id} (will skip duplicate metrics)"
                )

            # All tasks are processed as unified workflows
            logger.info("✅ Using unified workflow mode (all tasks processed as workflows)")

            # Generate a unique task ID
            task_id = str(uuid.uuid4())

            # Log task submission
            logger.info(
                f"🚀 Workflow task {task_id} submitted with {len(elements)} elements for URL: {url}"
            )
            logger.info("   Processing mode: Unified workflow (single Agent session)")
            logger.info(
                f"📝 User query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}"
            )

            # Atomically check capacity and submit — prevents TOCTOU race where
            # concurrent requests both pass a separate count check then both submit.
            accepted = task_processor.try_submit_task(
                config.max_concurrent_tasks,
                task_id,
                process_workflow_task,
                task_id,
                elements,
                url,
                user_query,
                session_config,
                enable_custom_actions,
                task_processor,
                parent_workflow_id,  # Pass parent_workflow_id to prevent duplicate metrics
                org_id,
                user_id,
            )
            if not accepted:
                active_count = task_processor.count_active_tasks()
                return jsonify(
                    {
                        "status": "busy",
                        "message": (
                            f"Service is at capacity ({active_count}/{config.max_concurrent_tasks} "
                            f"active tasks). Please try again later."
                        ),
                        "active_tasks": active_count,
                        "max_tasks": config.max_concurrent_tasks,
                    }
                ), 429

            # Return the task ID immediately
            return jsonify(
                {
                    "status": "processing",
                    "task_id": task_id,
                    "message": f"Workflow task submitted with {len(elements)} elements (unified session)",
                    "elements_count": len(elements),
                    "mode": "workflow",
                }
            ), 202

        except Exception as e:
            logger.error(f"Error in workflow submit endpoint: {e}", exc_info=True)
            return format_error_response(INTERNAL_ERROR_MESSAGE, 500)

    @app.route("/query/<task_id>", methods=["GET"])
    def query(task_id: str):
        """Query the status of a specific task."""
        try:
            task = task_processor.get_task_status(task_id)

            if task is None:
                return format_error_response("Task ID not found.", 404)

            status = task.get("status")

            # Format response based on status
            if status == "processing":
                response = format_task_response(task, include_results=False, truncate_objective=200)
                return jsonify(response), 202

            elif status == "running":
                response = format_task_response(task, include_results=False, truncate_objective=200)
                return jsonify(response), 202

            elif status == "completed":
                response = format_task_response(task, include_results=True, truncate_objective=200)
                logger.info(
                    f"Task {task_id} query completed: {task.get('results', {}).get('success', False)}"
                )
                return jsonify(response), 200

            else:
                response = format_task_response(task, include_results=False, truncate_objective=200)
                return jsonify(response), 200

        except Exception as e:
            logger.error(f"Error in query endpoint: {e}", exc_info=True)
            return format_error_response(INTERNAL_ERROR_MESSAGE, 500)

    @app.route("/tasks", methods=["GET"])
    def list_tasks():
        """List all tasks with their status."""
        try:
            all_tasks = task_processor.list_tasks()
            response = format_task_list_response(all_tasks)
            return jsonify(response), 200

        except Exception as e:
            logger.error(f"Error in list_tasks endpoint: {e}", exc_info=True)
            return format_error_response(INTERNAL_ERROR_MESSAGE, 500)

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify(
            {
                "status": "error",
                "message": "Endpoint not found",
                "available_endpoints": [
                    "GET / - Service info",
                    "GET /health - Health check",
                    "GET /probe - Legacy health check",
                    "POST /workflow - Process workflow task (RECOMMENDED)",
                    "POST /batch - Deprecated alias for /workflow",
                    "GET /query/<task_id> - Query task",
                    "GET /tasks - List tasks",
                ],
            }
        ), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors.

        The cause goes to the log, not to the caller. `str(error)` here is only
        werkzeug's canned InternalServerError text, so returning it told the
        client nothing while leaving the real exception unlogged.
        """
        logger.error(f"Unhandled 500 error: {error}", exc_info=True)
        return jsonify({"status": "error", "message": INTERNAL_ERROR_MESSAGE}), 500
