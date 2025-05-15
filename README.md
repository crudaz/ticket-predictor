# Ticket Prioritization MCP Server

A Flask-based API server for machine learning-based ticket prioritization. This service trains models on ticket data and provides predictions to help prioritize tickets efficiently.

## Features

- **Model Training**: Train a machine learning model on your ticket dataset
- **Ticket Prioritization**: Predict priority levels for new tickets
- **Containerized**: Easy deployment with Docker
- **RESTful API**: Simple HTTP endpoints for integration

## Installation

### Prerequisites

- Docker installed on your system

### Setup

1. Clone this repository
2. Navigate to the project directory
3. Build the Docker image:

```bash
docker build -t ticket-mcp-server .
```

## Usage

### Running the Server

Start the server with:

```bash
docker run -p 5001:5000 --name ticket-server ticket-mcp-server
```

This command:
- Maps port 5001 on your host to port 5000 in the container
- Names the container "ticket-server"
- Uses the "ticket-mcp-server" image

The server will be accessible at http://localhost:5001

### API Endpoints

#### Health Check

- **URL**: `/`
- **Method**: GET
- **Response**: Confirmation that the server is running

#### Training the Model

- **URL**: `/train`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameter**: `file` (CSV)
- **Response**: Status message indicating training success or failure

#### Making Predictions

- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameter**: `file` (CSV)
- **Response**: CSV file with predictions (named `tickets_priorizados.csv`)

## Data Format

### Training Data CSV

The CSV file for training should include these columns:
- `title`: Ticket title
- `description`: Ticket description
- `board_status_name`: Current status of the ticket
- `state_name`: Current state of the ticket
- `limit_date`: Due date (optional)
- Priority level field (target variable)

### Prediction Input CSV

The CSV file for prediction should include these columns:
- `title`: Ticket title
- `description`: Ticket description
- `board_status_name`: Current status of the ticket
- `state_name`: Current state of the ticket
- `limit_date`: Due date (optional)

### Prediction Output

The prediction API returns the input data with an additional column:
- `predicted_priority_level`: The model's priority prediction

## Data and Model Storage

- Models are stored in `/app/models` within the container
- Training data and prediction results are stored in `/app/data`
- Uploaded files are temporarily stored in `/app/uploads`

## Development

The server consists of two main components:
- `app.py`: Flask API implementation
- `ticket_model_api.py`: Core machine learning logic

## License

[Add your license information here]
