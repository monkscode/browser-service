"""
Basic usage example for browser-service package
"""
import asyncio
from browser_service.config import config
from browser_service.tasks import process_workflow_task


async def main():
    # Configure the service
    config.GEMINI_API_KEY = "your-api-key-here"
    config.ROBOT_LIBRARY = "browser"
    
    # Example workflow
    workflow = {
        "url": "https://example.com",
        "steps": [
            {"action": "click", "element": "Login button"},
            {"action": "type", "element": "Username field", "text": "user@example.com"}
        ]
    }
    
    # Process the workflow
    result = await process_workflow_task(workflow)
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
