import numpy as np
from flask import render_template, request, jsonify, Flask
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import pickle
import os
from flask import url_for

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	form = request.form.to_dict()
	

	return render_template('index.html',
	prediction_text="The probabilities of the positive tenancy and EET outcomes are {} and {}.".format(90, 2),
	hh='yes', complexity_score="Case Complexity : {} with Complexity Score : {}".format(70, 3))


@app.route('/starindex', methods=['GET', 'POST'])
def starindex():
	form = request.form.to_dict()
	
	return render_template('starindex.html')


@app.route('/predictstar', methods=['GET', 'POST'])
def predictstar():
	form = request.form.to_dict()
	

	return render_template('starindex.html',
	prediction_text="The probabilities of the positive tenancy and EET outcomes are {} and {}.".format(90, 2),
	hh='yes', complexity_score="Case Complexity : {} with Complexity Score : {}".format(70, 3))



@app.route('/report', methods=['GET', 'POST'])
def report():	
	return render_template('report.html')



@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)



if __name__=="__main__":
	app.run(debug=True)
