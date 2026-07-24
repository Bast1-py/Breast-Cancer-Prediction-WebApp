from flask import Flask


app = Flask(__name__)

# URL Binding
@app.route('/')
def hello():
    return ("I am SPARTA!!")

@app.route("/page2")
def hello_2():
    return("I am Admin!!")

@app.route("/admin")
def hello_3():
    return("This is a restricted page!!")

# app.run(debug=True)
app.run(port=5000)