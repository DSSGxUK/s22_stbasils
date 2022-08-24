# Improving Services for Homeless Youth in the UK

## Goal

There are around **86,000** young people experiencing homelessness in the UK and around **8,500** in the West Midlands. Because individuals have different levels of care needs, we must first understand how complex each case is, in order to plan and reserve care resources for them. 

There is a need for a simple metric to quantify the severity of issues, in order to provide more assistance to those who need it.

We developed a “complexity score” of Low, Medium, and High, which can be estimated at intake from an individual's background.

## Background

St Basils is a UK-based organisation focused on helping young people at risk of homelessness to prevent homelessness, find a home, build their confidence, and develop their skills.

## Project structure

//photo of the structure -- will be generated after we have all the files in the repo


## Project files description

* [models](https://github.com/DSSGxUK/s22_stbasils/tree/main/models) - the folder contains script defining the models, from their training to their deployment

    - [employability_outcome_model](https://github.com/DSSGxUK/s22_stbasils/tree/main/models/employability_outcome_model) - model which predicts the Education, Employment, or Training (EET) outcome
    - [tenancy_outcome_model](https://github.com/DSSGxUK/s22_stbasils/tree/main/models/tenancy_outcome_model) - model which predicts the accomodation outcome
    - [star_outcome_model](https://github.com/DSSGxUK/s22_stbasils/tree/main/models/star_outcome_model) - model which predicts the star score outcome
    
* [dashboard](https://github.com/DSSGxUK/s22_stbasils/tree/main/dashboard) - the folder containing the back-end and the front-end of the dashboard
    - [app.py](https://github.com/DSSGxUK/s22_stbasils/tree/main/dashboard/app.py) - python script to start a server on which the dashboard is displayed

* [causal_inference](https://github.com/DSSGxUK/s22_stbasils/tree/main/causal_inference) - the folder with causal inference tools and final application

## Usage




## Results

The result of the project is the dashboard with the tools helping to estimate the complexity score of the incoming young person and the relationship between outcome and provided support.

The dashboard consists of several sections:
1. **Accomodation & EET***: estimate the accomodation and EET outcomes, use them to estimate the complexity score of the incoming young person. The page also contains descriptive plots, which show the relationship between the positive outcomes and different support types.

2. **Star score**: estimate the final star score metric (an internal metric specific to ST Basils).

3. **Causal inference app**: assess the effect of support on the likelihood of positive outcome.

4. **Report**: internal report for St Basils with insights and findings after exploratory data analysis and model development.

## Contributors

- [Alina Voronina](https://github.com/linvieson)
- [Aryan Verma](https://github.com/infoaryan)
- [Hannah Olson-Williams](https://github.com/hannaheow)