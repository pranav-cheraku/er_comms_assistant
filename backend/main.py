# Main driver


# flask endpoints, call chatbot_intake class and doctor_summary class modules

# frontend info -> chatbot_intake -> summarize -> frontend

# record speech -> forward to doctor_summary module -> speech to text, extraction, text simplification -> output to frontend -> send email

from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/')
def driver():
    print('hello')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)