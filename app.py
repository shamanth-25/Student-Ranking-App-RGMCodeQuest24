# app.py

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib  # To load the saved model

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('random_forest_model.pkl')

# Define a route for ranking students
@app.route('/rank_students', methods=['POST'])
def rank_students():
    try:
        # Get student data from the request
        data = request.json

        # Convert the received data into a DataFrame
        student_df = pd.DataFrame(data)

        # Ensure the input features are in the correct order
        features = ['GPA', 'hackathon_participation', 'research_papers',
                    'teacher_assistance', 'event_organizing', 
                    'extracurricular_activities', 'consistency_score']

        # Make predictions using the model
        scores = model.predict(student_df[features])

        # Add scores to the dataframe
        student_df['ranking_score'] = scores

        # Sort the students by ranking score (descending)
        ranked_students = student_df.sort_values(by='ranking_score', ascending=False)

        # Convert the DataFrame to JSON and return the ranked students
        return jsonify(ranked_students.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
