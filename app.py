from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                day_type=request.form.get('day_type'),
                work_hours=float(request.form.get('work_hours')),
                screen_time_hours=float(request.form.get('screen_time_hours')),
                meetings_count=int(request.form.get('meetings_count')),
                breaks_taken=int(request.form.get('breaks_taken')),
                after_hours_work=int(request.form.get('after_hours_work')),
                sleep_hours=float(request.form.get('sleep_hours')),
                task_completion_rate=float(request.form.get('task_completion_rate'))
            )
            
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results, burnout_levels = predict_pipeline.predict(pred_df)
            
            return render_template(
                'home.html', 
                results=f"{results[0]:.2f}", 
                burnout_level=burnout_levels[0]
            )
        except Exception as e:
            return render_template('home.html', error=str(e))

if __name__ == "__main__":
    app.run(port=8000,debug=True)