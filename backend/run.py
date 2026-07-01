from app import create_app

app = create_app()

if __name__ == "__main__":
    print("SA Tax Chatbot API - listening on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
