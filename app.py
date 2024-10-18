from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = joblib.load('model/random_forest_model.pkl')

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle form submission and calculate ranking
@app.route('/rank', methods=['POST'])
def rank():
    if request.method == 'POST':
        # Retrieve form data
        gpa = float(request.form['GPA'])
        hackathon_participation = float(request.form['hackathon_participation'])
        research_papers = float(request.form['research_papers'])
        contributions = float(request.form['contributions'])

        # Prepare the data as a numpy array for prediction
        student_data = np.array([[gpa, hackathon_participation, research_papers, contributions]])

        # Predict the total score using the pre-trained model
        total_score = model.predict(student_data)[0]

        # Pass the total score to the results page
        return render_template('results.html', score=total_score)

if __name__ == '__main__':
    app.run(debug=True)
