
import json
from flask import Response, jsonify, render_template, request, stream_with_context

from .app import app
#from .chat_api import call_chat
from .chat_langchain import call_chat

demo_name = "RAG Chatbot"

@app.route("/")
def index():
    return render_template("index.html", demo_name=demo_name)

@app.route("/chat", methods=['POST'])
def chat_handler():
    request_message = request.json["message"]

    @stream_with_context
    def response_stream():
        for chunk in call_chat(request_message):
            # returning a json format for easier encoding
            # each chunk {"token": "..."}
            yield json.dumps(chunk, ensure_ascii=False) + "\n"

    return Response(response_stream(), mimetype="text/event-stream")

@app.route("/user/<user_id>", methods=["GET"])
def get_user(user_id):
    # Retrieve user data from the database and return a JSON response
    return {"name": "Example Name"}
from .analytics import M&AAnalytics

# Initialize analytics
df = pd.read_csv("./src/Tech M&A Deals (1988-2021).csv")
analytics = M&AAnalytics(df)

@app.route("/api/company/<company_name>", methods=['GET'])
def get_company_history(company_name):
    """Get acquisition history for a specific company"""
    history = analytics.get_company_acquisition_history(company_name)
    return jsonify(history.to_dict('records'))

@app.route("/api/trends", methods=['GET'])
def get_trends():
    """Get M&A trends analysis"""
    trends = analytics.get_acquisition_trends()
    return jsonify(trends.to_dict())

@app.route("/api/top-deals", methods=['GET'])
def get_top_deals():
    """Get largest M&A deals"""
    n = request.args.get('n', 10, type=int)
    deals = analytics.get_largest_deals(n)
    return jsonify(deals.to_dict('records'))
