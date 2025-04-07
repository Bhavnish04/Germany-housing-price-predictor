from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')  # Load the trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define expected feature names
        feature_names = [
            'meal_inexpensive', 'cappuccino', 'bread', 'milk', 'eggs', 'cheese',
            'apt_1bed_city', 'monthly_pass', 'internet'
        ]
        
        features = []
        for name in feature_names:
            value = request.form.get(name)
            if value is None or value.strip() == "":
                return render_template('index.html', prediction_text="⚠️ Please fill all fields correctly.")
            features.append(float(value))

        final_input = np.array(features).reshape(1, -1)
        prediction = model.predict(final_input)[0]
        return render_template('index.html', prediction_text=f'Predicted Housing Price: €{round(prediction, 2)}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
