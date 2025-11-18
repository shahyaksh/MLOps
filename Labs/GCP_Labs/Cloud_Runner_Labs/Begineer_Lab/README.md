# Cloud Run Flask Application

A simple Flask application designed to be deployed on Google Cloud Run. This application demonstrates basic REST API endpoints and serves as a beginner-friendly example for Cloud Run deployment.

## Features

- Simple Flask web application
- Multiple REST API endpoints for testing
- Health check endpoint for monitoring
- Containerized with Docker
- Ready for Cloud Run deployment

## API Endpoints

### 1. Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a simple "Hello, World!" message
- **Response**: `"Hello, World!"`

### 2. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Health check endpoint for Cloud Run monitoring
- **Response**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2025-01-XX...",
    "service": "Cloud Run Flask App"
  }
  ```

### 3. API Information
- **URL**: `/api/info`
- **Method**: `GET`
- **Description**: Returns information about all available endpoints
- **Response**:
  ```json
  {
    "message": "This is a test endpoint for Cloud Run",
    "endpoints": {
      "/": "Hello World endpoint",
      "/health": "Health check endpoint",
      "/api/info": "API information endpoint"
    },
    "deployment": "Cloud Run",
    "timestamp": "2025-01-XX..."
  }
  ```

### 4. Test Endpoint
- **URL**: `/api/test`
- **Method**: `GET`
- **Description**: Simple test endpoint to verify Cloud Run functionality
- **Response**:
  ```json
  {
    "success": true,
    "message": "Cloud Run test endpoint is working!",
    "data": {
      "test_id": "test_001",
      "status": "active"
    }
  }
  ```

## Prerequisites

- Python 3.8 or higher
- Docker (for containerization)
- Google Cloud SDK (`gcloud`) installed and configured
- A Google Cloud Project with Cloud Run API enabled

## Local Development

### 1. Install Dependencies

```bash
pip install flask
```

### 2. Run the Application Locally

```bash
python app.py
```

The application will be available at `http://localhost:8080`

### 3. Test Endpoints Locally

```bash
# Test root endpoint
curl http://localhost:8080/

# Test health check
curl http://localhost:8080/health

# Test API info
curl http://localhost:8080/api/info

# Test endpoint
curl http://localhost:8080/api/test
```

## Docker Build and Test

### Build Docker Image

```bash
docker build -t cloud-run-flask-app .
```

### Run Docker Container Locally

```bash
docker run -p 8080:8080 cloud-run-flask-app
```

## Deployment to Google Cloud Run

### Step 1: Authenticate and Configure Google Cloud

1. **Initialize gcloud** (if not already done):
   ```bash
   gcloud init
   ```
   - Select your Google account
   - Choose your Google Cloud project (or create a new one)
   - Optionally set default Compute Engine region/zone

2. **Verify your configuration**:
   ```bash
   gcloud config list
   ```

3. **Enable required APIs** (if not already enabled):
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

### Step 2: Deploy to Cloud Run (Simplified Method)

The easiest way to deploy is using `gcloud run deploy --source .` which automatically builds and deploys your application:

```bash
gcloud run deploy --source .
```

**During the deployment process, you'll be prompted for:**
- **Service name**: Enter a name for your service (e.g., `first-flask-app`)
- **Region**: Select a region (e.g., `us-east4`, `us-central1`, etc.)
- **Allow unauthenticated invocations**: Type `y` to make the service publicly accessible

**Example output:**
```
Service name (begineerlab): first-flask-app
Please specify a region: [select from list]
Allow unauthenticated invocations to [first-flask-app] (y/N)? y

Building using Dockerfile and deploying container to Cloud Run service...
✓ Building and deploying new service... Done.
Service URL: https://first-flask-app-XXXXX.us-east4.run.app
```

**Note**: The first time you deploy, Cloud Run will automatically create an Artifact Registry repository named `cloud-run-source-deploy` in your selected region to store the built container.


### Step 3: Test the Deployed Application

Once deployed, Cloud Run will provide a service URL in the format:
`https://SERVICE_NAME-PROJECT_NUMBER.REGION.run.app`

**Example**: `https://first-flask-app-66725207998.us-east4.run.app`

Test all the endpoints:

```bash
# Replace with your actual Cloud Run service URL
SERVICE_URL="https://first-flask-app-66725207998.us-east4.run.app"

# Test root endpoint
curl $SERVICE_URL/

# Test health check
curl $SERVICE_URL/health

# Test API info
curl $SERVICE_URL/api/info

# Test endpoint
curl $SERVICE_URL/api/test
```

Or test directly in your browser by visiting the URLs:
- `https://YOUR_SERVICE_URL/`
- `https://YOUR_SERVICE_URL/health`
- `https://YOUR_SERVICE_URL/api/info`
- `https://YOUR_SERVICE_URL/api/test`

## Project Structure

```
Begineer_Lab/
├── app.py          # Flask application with all endpoints
├── Dockerfile      # Docker configuration for containerization
└── README.md       # This file
```

## Configuration

The application runs on:
- **Host**: `0.0.0.0` (required for Cloud Run)
- **Port**: `8080` (Cloud Run default)

## Monitoring

After deployment, you can monitor your Cloud Run service:
- View metrics in the Cloud Run Console
- Check request logs in Cloud Logging
- Monitor performance metrics (latency, request count, memory usage)

## Auto-Scaling

Cloud Run automatically scales your service based on incoming traffic:
- Scales to zero when there's no traffic
- Scales up automatically when traffic increases
- Configure min/max instances in the Cloud Run settings if needed

## Troubleshooting

- **Port 8080**: Ensure your app listens on port 8080 (Cloud Run requirement)
- **Host 0.0.0.0**: The app must bind to `0.0.0.0`, not `127.0.0.1`
- **Health Checks**: Use the `/health` endpoint for Cloud Run health checks
- **Logs**: Check Cloud Logging for any runtime errors

## License

This is a learning project for MLOps coursework.
