"""
WebSocket API Routes
Real-time progress updates and streaming
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.document_subscribers: Dict[int, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from document subscriptions
        for doc_id, subscribers in self.document_subscribers.items():
            if client_id in subscribers:
                subscribers.remove(client_id)
        
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {str(e)}")
                self.disconnect(client_id)
    
    async def broadcast_document_update(self, document_id: int, message: dict):
        """Broadcast update to all clients subscribed to a document"""
        subscribers = self.document_subscribers.get(document_id, [])
        for client_id in subscribers.copy():  # Copy to avoid modification during iteration
            await self.send_personal_message(message, client_id)
    
    def subscribe_to_document(self, client_id: str, document_id: int):
        """Subscribe client to document updates"""
        if document_id not in self.document_subscribers:
            self.document_subscribers[document_id] = []
        
        if client_id not in self.document_subscribers[document_id]:
            self.document_subscribers[document_id].append(client_id)
            logger.info(f"Client {client_id} subscribed to document {document_id}")

manager = ConnectionManager()

@router.websocket("/progress/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time progress updates
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "subscribe_document":
                document_id = message.get("document_id")
                if document_id:
                    manager.subscribe_to_document(client_id, document_id)
                    await manager.send_personal_message({
                        "type": "subscription_confirmed",
                        "document_id": document_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }, client_id)
            
            elif message.get("type") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, client_id)
            
            elif message.get("type") == "get_status":
                document_id = message.get("document_id")
                if document_id:
                    # In a real implementation, get actual status from database
                    await manager.send_personal_message({
                        "type": "status_update",
                        "document_id": document_id,
                        "status": "processing",
                        "progress": 75.0,
                        "current_step": "Analyzing patterns",
                        "timestamp": datetime.utcnow().isoformat()
                    }, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        manager.disconnect(client_id)

# Helper functions for sending updates (to be called from other parts of the application)
async def send_processing_update(document_id: int, status: str, progress: float, step: str):
    """Send processing update to all subscribers of a document"""
    message = {
        "type": "processing_update",
        "document_id": document_id,
        "status": status,
        "progress": progress,
        "current_step": step,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast_document_update(document_id, message)

async def send_analysis_complete(document_id: int, results_summary: dict):
    """Send analysis completion notification"""
    message = {
        "type": "analysis_complete",
        "document_id": document_id,
        "results_summary": results_summary,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast_document_update(document_id, message)

async def send_error_notification(document_id: int, error_message: str):
    """Send error notification"""
    message = {
        "type": "error",
        "document_id": document_id,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast_document_update(document_id, message)

# Export the manager for use in other modules
__all__ = ["manager", "send_processing_update", "send_analysis_complete", "send_error_notification"]