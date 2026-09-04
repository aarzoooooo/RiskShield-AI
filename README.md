# RiskShield-AI

AI-powered Account Takeover Risk Detection and Security Assessment.

## Overview

RiskShield-AI is a machine learning-based security application designed to identify potentially suspicious login attempts and assess the risk of account takeover.

The system analyzes login and device-related information such as IP address, login status, device type, operating system, browser information, geographic information, and round-trip time to determine whether a login attempt is potentially risky.

The application provides both:

-  Single Login Analysis
-  Batch CSV Analysis

through an interactive Streamlit dashboard.

---

##  Problem Statement

Account takeover attacks occur when attackers gain unauthorized access to legitimate user accounts.

Traditional security systems often rely on fixed rules and may struggle to identify unusual combinations of login behavior and device characteristics.

RiskShield-AI uses machine learning to analyze login patterns and identify potentially suspicious activity, helping security teams prioritize risky login attempts.

---

##  Solution

RiskShield-AI uses a supervised machine learning model trained on login activity data.

The system:

1. Accepts login information from a user or CSV dataset.
2. Performs feature processing and category encoding.
3. Passes the processed information to the trained ML model.
4. Calculates the probability of account takeover.
5. Compares the probability with a risk threshold.
6. Classifies the login as normal or potentially risky.
7. Provides risk-related information through the Streamlit interface.

---

##  Key Features

###  Single Login Analysis

Users can enter individual login information including:

- Round-Trip Time
- User Agent String
- IP Address
- Browser Name and Version
- Country
- Region
- City
- OS Name and Version
- Device Type
- Login Successful status
- Is Attack IP status

The application then evaluates the login attempt and provides a risk assessment.

###  Batch Analysis

Users can upload a CSV file containing multiple login records.

The system processes the records and performs risk analysis across the entire dataset.

This makes the application useful for analyzing large numbers of login events.

###  Machine Learning Detection

The trained model identifies patterns associated with account takeover attempts instead of relying only on manually defined rules.

###  Fast Risk Assessment

The trained model is loaded directly for prediction, allowing new login attempts to be evaluated without retraining the model.

---

##  How It Works

Login Data
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Trained ML Model
     ↓
Risk Probability
     ↓
Threshold Comparison
     ↓
Risk Assessment

## Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Application
- Streamlit

### Data Processing
- Feature Engineering
- Categorical Encoding
- Boolean Feature Conversion

### Deployment
- Streamlit Cloud
