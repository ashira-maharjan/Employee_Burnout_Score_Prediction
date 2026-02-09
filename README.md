
# Employee Burnout Score Prediction

This project is a **Machine Learning application** designed to **predict the burnout score** of employees based on workplace and personal factors. The goal is to help organizations identify employees at risk of burnout and take proactive steps to support their well-being.

---

##  Table of Contents

*  About
*  Features
*  Tech Stack
*  Project Structure
*  Installation
*  Usage



---

##  About

Employee burnout is a common workplace challenge that affects productivity, mental health, and job satisfaction. This project trains a machine learning model to estimate an employee’s burnout score from relevant features in the dataset. Based on this score, organizations can better understand risk patterns and plan interventions.

---

##  Features

*  Data preprocessing and cleaning
*  Exploratory Data Analysis (EDA)
*  Training a Machine Learning model to predict burnout score
*  Evaluation with performance metrics
*  (Optional) Web interface for running predictions locally

---

## 🛠️ Tech Stack

The project uses the following tools and libraries:

* Python
* Machine Learning libraries (e.g., Scikit-Learn, CatBoost)
* Data manipulation (Pandas, NumPy)
* (Optional) Flask / Streamlit for interface
* Jupyter Notebook for experimentation

---

##  Project Structure

```plaintext
📦 Employee_Burnout_Score_Prediction
 ┣ 📂 notebook        # Jupyter notebooks used for experimentation
 ┣ 📂 src             # Source code for model training and utilities
 ┣ 📂 templates       # (If web app) HTML template files
 ┣ 📜 app.py          # (If web app) Runs the prediction interface
 ┣ 📜 train.py        # Script to train the machine learning model
 ┣ 📜 requirements.txt# Python dependencies
 ┣ 📜 README.md       # Project documentation
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ashira-maharjan/Employee_Burnout_Score_Prediction.git
cd Employee_Burnout_Score_Prediction
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Usage

###  Train the Model

To train the burnout prediction model from scratch:

```bash
python train.py
```

### 🖥️ Run Local Web App (if available)

```bash
python app.py
```

Open your browser and go to:

```text
http://localhost:8000
```

Follow the on-screen steps to input employee data and get a predicted burnout score.

---



