from flask import Flask, request, jsonify, render_template
from chatbot import ChatBot
from flask_cors import CORS
import os
from chatbot import ChatBot

app = Flask(__name__)
CORS(app)

# Initialize the chatbot
#chatbot = ChatBot()

# If your HTML is in /views folder, you can serve it like this
@app.route('/')
def load_chatbot():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/test', methods=['GET'])
def test():
    return "Server is up!",200


@app.route('/getResponse', methods=['POST'])
def get_response():
    """
    API endpoint that receives a message, calls the chatbot, and returns the response.
    Expects a POST request with a JSON body containing the 'message'.
    """
    # Ensure the request contains JSON data
    if request.is_json:
        # Parse the message from the request body
        data = request.get_json()
        message = data.get('message')

        # Check if message is provided
        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Call the chatbot to get a response
        try:
            response = ChatBot.get_response_inference(message)
            return jsonify({"response": response}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Return an error if the request body is not JSON
    return jsonify({"error": "Request body must be JSON"}), 400


if __name__ == "__main__":
    # Get the PORT from environment variables, default to 5000 if not set
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port)
