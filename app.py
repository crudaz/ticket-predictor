# -*- coding: utf-8 -*-
"""
Flask API server to expose ticket prioritization model training and prediction.
"""

import os
from flask import Flask, request, jsonify, send_file, Response
import ticket_model_api  # Import the refactored model logic
from werkzeug.utils import secure_filename
import traceback  # For detailed error logging

# Configuration
# Folder inside the container to temporarily store uploads
UPLOAD_FOLDER = '/app/uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * \
    1024  # Optional: Limit upload size (e.g., 16MB)


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Basic index route to confirm the server is running."""
    return jsonify({"message": "MCP Server for Ticket Prioritization is running."})


@app.route('/train', methods=['POST'])
def train_endpoint():
    """
    API endpoint to train the model.
    Expects a POST request with a CSV file named 'file'.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save the uploaded file temporarily
        train_data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(train_data_path)
            print(f"Training file saved to {train_data_path}")

            # Call the training function from the imported module
            success, message = ticket_model_api.perform_training(
                train_data_path)

            # Clean up the uploaded file after training
            os.remove(train_data_path)
            print(f"Removed temporary training file: {train_data_path}")

            if success:
                # Send the success message required by Claude Desktop
                return jsonify({"message": "The model have been trained"}), 200, {'Content-Type': 'application/json'}
            else:
                return jsonify({"error": f"Training failed: {message}"}), 500

        except Exception as e:
            # Log the full traceback for debugging
            print(
                f"An error occurred during training: {traceback.format_exc()}")
            # Clean up if an error occurred during processing
            if os.path.exists(train_data_path):
                os.remove(train_data_path)
            return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed. Please upload a CSV file."}), 400


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    API endpoint to perform predictions using the trained model.
    Expects a POST request with a CSV file named 'file'.
    Returns the prediction results as a CSV file.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save the uploaded file temporarily
        predict_data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(predict_data_path)
            print(f"Prediction file saved to {predict_data_path}")

            # Call the prediction function from the imported module
            success, result_msg_or_path = ticket_model_api.perform_prediction(
                predict_data_path)

            # Clean up the uploaded file after prediction
            os.remove(predict_data_path)
            print(f"Removed temporary prediction file: {predict_data_path}")

            if success:
                output_csv_path = result_msg_or_path
                # Check if the output file exists before sending
                if os.path.exists(output_csv_path):
                    # Send the generated CSV file back to the client
                    print(f"Sending prediction file: {output_csv_path}")
                    return send_file(
                        output_csv_path,
                        mimetype='text/csv',
                        download_name='tickets_priorizados.csv',  # The required output filename
                        as_attachment=True
                    )
                else:
                    print(
                        f"Error: Predicted file not found at {output_csv_path}")
                    return jsonify({"error": "Prediction succeeded but the output file was not found."}), 500
            else:
                # Prediction failed, result_msg_or_path contains the error message
                return jsonify({"error": f"Prediction failed: {result_msg_or_path}"}), 500

        except Exception as e:
            # Log the full traceback for debugging
            print(
                f"An error occurred during prediction: {traceback.format_exc()}")
            # Clean up if an error occurred during processing
            if os.path.exists(predict_data_path):
                os.remove(predict_data_path)
            return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed. Please upload a CSV file."}), 400


if __name__ == '__main__':
    # Run the Flask app
    # Host '0.0.0.0' makes it accessible from outside the container
    # Turn debug off for production/container use
    app.run(host='0.0.0.0', port=5000, debug=False)
