from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/health')
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Cloud Run Flask App"
    }), 200

@app.route('/api/info')
def api_info():
    """API info endpoint to test Cloud Run"""
    return jsonify({
        "message": "This is a test endpoint for Cloud Run",
        "endpoints": {
            "/": "Hello World endpoint",
            "/health": "Health check endpoint",
            "/api/info": "API information endpoint"
        },
        "deployment": "Cloud Run",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/test')
def test_endpoint():
    """Simple test endpoint"""
    return jsonify({
        "success": True,
        "message": "Cloud Run test endpoint is working!",
        "data": {
            "test_id": "test_001",
            "status": "active"
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)