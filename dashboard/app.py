import numpy as np
import os
from flask import render_template, request, jsonify, Flask, url_for
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

###########################################
### Load the models in global variables ###
###########################################
output = open('model_tenancy/model_final.pkl','rb')
best_model_tenancy = pickle.load(output)
output.close()
output = open('model_eet/model.pkl','rb')
best_model_eet = pickle.load(output)
output.close()


@app.route('/')
def home():
	return render_template('index.html')
	
@app.route('/starindex', methods=['GET', 'POST'])
def starindex():
	form = request.form.to_dict()
	return render_template('starindex.html')

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

@app.route('/predict', methods=['POST'])
def predict():
	form = request.form.to_dict()
	
	# Check if H1, I1, Medical Issue, G1 variables are in the form or not
	# And replace the empty values with numpy.nan
	form = checkVariables(form)
	form = handleNulls(form)
	
	#Generate Predicitions
	probab_tenancy, probab_eet = generatePredictions(form)
	
	#Generate all Graphs
	generateAndSaveGraphs(form)
	
	#Calculate the Complexity Score
	complex_probability, complexity = calculateComplexity(form['mixer'], probab_tenancy, probab_eet)
	
	#render the template
	return render_template('index.html',
	prediction_text="The probability of the tenancy and the neet outcome is {} and {}".format(round(probab_tenancy,2),round(probab_eet,2)),
	showOutputs='yes', complexity_score="Case Complexity : {} with Complexity Score : {}".format(complexity, round(complex_probability,2)), low_ten=int(probab_tenancy*100-3.9), low_eet=int(probab_eet*100-4.0))

def checkVariables(form):
	# This function checks if the required variables are coming from the HTML form
	# If the variables (Which are in form of checkboxes) are not there, include them
	#################################################################
	########## Varibles Checked -> H1, I1, Medical Issue, G1 ########
	#################### Default Value -> 'No' ######################
	#################################################################
	if('H1' not in form):
		form['H1'] = 'No'
	if('I1' not in form):
		form['I1'] = 'No'
	if('Medical Issue' not in form):
		form['Medical Issue'] = 'No'
	if('G1' not in form):
		form['G1'] = 'No'
	return form

def handleNulls(form):
	# Checks if the form variables are containing null
	# The nulls are replaced with numpy.nan
	if(form['Nationality']==''):
		form['Nationality']=np.nan
	if(form['Sexual Orientation']==''):
		form['Sexual Orientation']=np.nan
	if(form['Marital Status']==''):
		form['Marital Status']=np.nan
	if(form['Acc Type prev']==''):
		form['Acc Type prev']=np.nan
	if(form['Gender']==''):
		form['Gender']=np.nan
	if(form['EET status']==''):
		form['EET status']=np.nan
	if(form['Religion']==''):
		form['Religion']=np.nan
	if(form['C1']==''):
		form['C1']=np.nan
	if(form['Preferred Language']==''):
		form['Preferred Language']=np.nan
	return form
	
def calculateComplexity(mixer, probab_tenancy, probab_eet):
	# mixer -> Float, determines the ratio in which tenancy and eet probabilties are combined to make complexity score
	############################################################################
	#### complexity score = mixer x probab_tenancy + (1-mixer) x probab_eet ####
	############################################################################
	
	mixing_probab = int(mixer)/100.0
	complex_probability = ((mixing_probab * probab_tenancy) + ((1-mixing_probab)* probab_eet)) /2.0
	if(complex_probability<0.33):
		complexity="Hard!"
	elif(complex_probability<0.66):
		complexity="Medium"
	else:
		complexity="Easy"
	return complex_probability, complexity

def generatePredictions(form):
	# This function generates the prediciton from both the models 
	# form -> a dictionary of all the inputs from the HTML form
	
	# Threshold is a value which defines class boundary
	# Threshold is 0.441
	# Ex. Above probability 0.441 the outcome will be +ve 
	transformed_input_tenants = transform_input_tenancy(form)
	probab_tenancy = best_model_tenancy.predict_proba(transformed_input_tenants)[0][1]
	
	# The thresh is 0.43
	# Ex. Above probability 0.43 the outcome will be -ve 
	transformed_input_eet = transform_input_eet(form)
	probab_eet = best_model_eet.predict_proba(transformed_input_eet)[0][0]
	
	return probab_tenancy, probab_eet

def generateAndSaveGraphs(form):
	# This funciton generates the graphs and saves them
	# The saved graphs can be displayed in the html form 
	image_url = ''
	make_graph_TimePerSession_eet(form)
	make_graph_TimePerSession_tenancy(form)
	make_graph_SessionPerWeek_eet(form)
	make_graph_SessionPerWeek_tenancy(form)
	make_graph_Scheme(form)

def transform_input_eet(x):
	# Input -> dictionary x, contains the information fromt he HTML form
	# Output -> Desired shape of the data which need to be fed in model
    transformed = []
    
    # Label transform the data 
    output = open('model_eet/Initial.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    value = x['EET status']
    if(value!='NEET' and value!='EET'):
        value='NEET'
    a = le.transform([value])
    transformed.append(a[0])
    
    output = open('model_eet/MedicalIssue.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    a = le.transform([x['Medical Issue']])
    transformed.append(a[0])
    
    output = open('model_eet/G1.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    a = le.transform([x['G1']])
    transformed.append(a[0])
    
    output = open('model_eet/H1.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    a = le.transform([x['H1']])
    transformed.append(a[0])
    
    output = open('model_eet/I1.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    a = le.transform([x['I1']])
    transformed.append(a[0])
    
    # Standard Scale the data
    output = open('model_eet/StandardScaler.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    a = le.transform([[x['TotalDisabilty'],x['TotalMentalHealth'],x['Time per Session'],x['Session per week']]])
    transformed.append(a[0][0])
    transformed.append(a[0][1])
    transformed.append(a[0][2])
    transformed.append(a[0][3])
    transformed = np.asarray(transformed)
    
    #One hot encoding the data finally
    output = open('model_eet/OneHotEncoder.pkl', 'rb')
    ohe = pickle.load(output)
    output.close()
    codes = ohe.transform([[x['Gender'],x['Preferred Language'],x['Nationality'],x['B1'],x['C1'],x['Marital Status'],x['Sexual Orientation'],x['Religion']]]).toarray()
    transformed = np.append(transformed,codes[0])
    transformed = np.reshape(transformed,(1,50))
    
    return transformed

def transform_input_tenancy(x):
	# Input -> dictionary x, contains the information fromt he HTML form
	# Output -> Desired shape of the data which need to be fed in model
    transformed = []
    
    # Standard Scale the data
    output = open('model_tenancy/StandardScaler.pkl', 'rb')
    le = pickle.load(output)
    output.close()
    a = le.transform([[x['TotalDisabilty'],x['TotalMentalHealth'],x['Time per Session'],x['Session per week']]])
    transformed.append(a[0][0])
    transformed.append(a[0][1])
    transformed.append(a[0][2])
    transformed.append(a[0][3])
    
    transformed = np.asarray(transformed)
    

    #One hot encoding the data finally
    output = open('model_tenancy/OneHotEncoder.pkl', 'rb')
    ohe = pickle.load(output)
    output.close()
    codes = ohe.transform([[x['Acc Type prev'],x['B1'],x['C1'],x['Economic Status'],x['Area'],x['Scheme'],x['EET status'],x['Service Type'],x['Religion']]]).toarray()
    transformed = np.append(transformed,codes[0])
    transformed = np.reshape(transformed,(1,72))
    
    return transformed
    
def make_graph_TimePerSession_eet(x):
    #########################################
    ########### TIME PER SESSION ############
    #########################################
    
	# This function creates the various values of Time per session
	# At all the values of time per session, probability of outcome is predicted
	# These probabilities are then plotted on y_axis
	# x_Axis contains the various values of time per session
    import random
    # Just the label for plot x-axis
    case = np.zeros(20, np.float64)
    f, ax = plt.subplots()
    samples = np.arange(0,len(case))
    # Generating colors for the plot of matplotlib line poots to be ploitte d
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(['1']))]
    color_count = 0
    count=0
    for s in np.arange(15.0,75.0,3.0):
        x1 = x.copy()
        x1['Session per week'] = s
        transformed = transform_input_eet(x1)
        predicted_probab = best_model_eet.predict_proba(transformed)[0][0]
        case[count] = predicted_probab
        count = count +1 
    ax.plot(samples, case, linestyle='--', marker='o', color=colors[color_count])
    ax.fill_between(samples, case-0.05, case+0.05 ,alpha=0.2, facecolor=colors[color_count])
    color_count = color_count+1
    labels = np.arange(15.0,75.0,3.0)
    labels = labels.astype('int8')
    ax.set_xticks(samples)
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.axhline(y=0.43, color='g', linestyle='-')
    #plot_width, plot_height = (24,18)
    #plt.rcParams['figure.figsize'] = (plot_width,plot_height)
    #plt.rcParams['font.size']=18
    plt.xlabel("Time per Session (Minutes)")
    plt.ylabel("Positive outcome Probability")
    plt.savefig('static/graphs/tps_eet.png')

def make_graph_TimePerSession_tenancy(x):
    #########################################
    ########### TIME PER SESSION ############
    #########################################
    
	# This function creates the various values of Time per session
	# At all the values of time per session, probability of outcome is predicted
	# These probabilities are then plotted on y_axis
	# x_Axis contains the various values of time per session
    import random
    # Just the label for plot x-axis
    case = np.zeros(20, np.float64)
    f, ax = plt.subplots()
    samples = np.arange(0,len(case))
    # Generating colors for the plot of matplotlib line poots to be ploitte d
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(['1']))]
    color_count = 0
    count=0
    for s in np.arange(15.0,75.0,3.0):
        x1 = x.copy()
        x1['Session per week'] = s
        transformed = transform_input_tenancy(x1)
        predicted_probab = best_model_tenancy.predict_proba(transformed)[0][1]
        case[count] = predicted_probab
        count = count +1
    ax.plot(samples, case, linestyle='--', marker='o', color=colors[color_count])
    ax.fill_between(samples, case-0.039, case+0.039 ,alpha=0.2, facecolor=colors[color_count])
    color_count = color_count+1
    labels = np.arange(15.0,75.0,3.0)
    labels = labels.astype('int8')
    ax.set_xticks(samples)
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.axhline(y=0.441, color='g', linestyle='-')
    #plot_width, plot_height = (24,18)
    #plt.rcParams['figure.figsize'] = (plot_width,plot_height)
    #plt.rcParams['font.size']=18
    plt.xlabel("Time per Session (Minutes)")
    plt.ylabel("Positive outcome Probability")
    plt.savefig('static/graphs/tps_tenancy.png')

def trunc(values, decs=0):
	return np.trunc(values*10**decs)/(10**decs)

def make_graph_SessionPerWeek_eet(x):
    #########################################
    ########### SESSION PER WEEK ############
    #########################################
    
	# This function creates the various values of Sessions per week
	# At all the values of Session per Week, probability of outcome is predicted
	# These probabilities are then plotted on y_axis
	# x_Axis contains the various values of Session per Week
    import random
    # Just the label for plot x-axis
    case = np.zeros(20, np.float64)
    f, ax = plt.subplots()
    samples = np.arange(0,len(case))
    # Generating colors for the plot of matplotlib line poots to be ploitte d
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(['1']))]
    color_count = 0
    count=0
    for s in np.arange(0.5,6.5,0.3):
        x1 = x.copy()
        x1['Time per Session'] = s
        transformed = transform_input_eet(x1)
        predicted_probab = best_model_eet.predict_proba(transformed)[0][0]
        case[count] = predicted_probab 
        count = count +1 
    ax.plot(samples, case, linestyle='--', marker='o', color=colors[color_count])
    ax.fill_between(samples, case-0.05, case+0.05 ,alpha=0.2, facecolor=colors[color_count])
    color_count = color_count+1
    labels = np.arange(0.5,6.5,0.3)
    labels = trunc(labels,1)
    ax.set_xticks(samples)
    ax.set_xticklabels(labels, rotation=90)
    ax.axhline(y=0.43, color='g', linestyle='-')
    #plot_width, plot_height = (24,18)
    #plt.rcParams['figure.figsize'] = (plot_width,plot_height)
    #plt.rcParams['font.size']=20
    #plt.legend()
    plt.xlabel("Session per Week")
    plt.ylabel("Positive outcome Probability")
    plt.savefig('static/graphs/spw_eet.png')

def make_graph_SessionPerWeek_tenancy(x):
    #########################################
    ########### SESSION PER WEEK ############
    #########################################
	
	# This function creates the various values of Sessions per week
	# At all the values of Session per Week, probability of outcome is predicted
	# These probabilities are then plotted on y_axis
	# x_Axis contains the various values of Session per Week
    import random
    # Just the label for plot x-axis
    case = np.zeros(20, np.float64)
    f, ax = plt.subplots()
    samples = np.arange(0,len(case))
    # Generating colors for the plot of matplotlib line poots to be ploitte d
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(['1']))]
    color_count = 0
    count=0
    for s in np.arange(0.5,6.5,0.3):
        x1 = x.copy()
        x1['Time per Session'] = s
        transformed = transform_input_tenancy(x1)
        predicted_probab = best_model_tenancy.predict_proba(transformed)[0][1]
        case[count] = predicted_probab
        count = count +1 
    ax.plot(samples, case, linestyle='--', marker='o', color=colors[color_count])
    ax.fill_between(samples, case-0.039, case+0.039 ,alpha=0.2, facecolor=colors[color_count])
    color_count = color_count+1
    labels = np.arange(0.5,6.5,0.3)
    labels = trunc(labels,1)
    ax.set_xticks(samples)
    ax.set_xticklabels(labels, rotation=90)
    ax.axhline(y=0.441, color='g', linestyle='-')
    #plot_width, plot_height = (24,18)
    #plt.rcParams['figure.figsize'] = (plot_width,plot_height)
    #plt.rcParams['font.size']=20
    #plt.legend()
    plt.xlabel("Session per Week")
    plt.ylabel("Positive outcome Probability")
    plt.savefig('static/graphs/spw_tenancy.png')

def make_graph_Scheme(x):
    #########################################
    ################ SCHEME #################
    #########################################
    
	# This function creates the various values of Scheme
	# At all the values of Scheme, probability of outcome is predicted
	# These probabilities are then plotted on y_axis
	# x_Axis contains the various Schemes
	
    import random
    # Just the label for plot x-axis
    case = np.zeros(15, np.float64)
    f, ax = plt.subplots()
    # Generating colors for the plot of matplotlib line poots to be ploitte d
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(['1']))]
    color_count = 0
    count=0
    schemes = ['AE', 'fill', 'CC', 'BD', 'CD', 'AL', 'AC', 'BM', 'BJ', 'AT', 'AB',
    'AU', 'AY', 'AM', 'AH']
    samples = np.arange(0,len(schemes))
    for s in schemes:
        x1 = x.copy()
        x1['Scheme'] = s
        transformed = transform_input_tenancy(x1)
        predicted_probab = best_model_tenancy.predict_proba(transformed)[0][1]
        case[count] = predicted_probab 
        count = count +1 
    #ax.plot(samples, case, linestyle='--', marker='o', color=colors[color_count])
    ax.errorbar(x=samples,y=case, yerr=0.039, color="#550E9C", capsize=3, linestyle='None', marker = 's', markersize=7, mfc='black', mec="black")
    #ax.fill_between(samples, case-0.039, case+0.039 ,alpha=0.2, facecolor=colors[color_count])
    ax.plot([schemes.index(x['Scheme'])], [case[schemes.index(x['Scheme'])]], 'rp', markersize=14)
    color_count = color_count+1
    ax.set_xticks(samples)
    ax.set_xticklabels(schemes, rotation=90)
    ax.axhline(y=0.44, color='g', linestyle='-')
    plt.xlabel("Scheme")
    plt.ylabel("Positive outcome Probability")
    plt.savefig('static/graphs/scheme.png')

if __name__=="__main__":
	app.run(debug=True)