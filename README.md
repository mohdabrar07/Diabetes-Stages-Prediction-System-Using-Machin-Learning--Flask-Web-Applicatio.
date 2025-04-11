# Diabetes-Stages-Prediction-System-Using-Machin-Learning--Flask-Web-Applicatio.
This is our Major Project for 8th sem, "Diabetes Stages Prediction System Using Machin Learning"- a Flask based Web Application, used to predict the different stages of diabetes using parameters like age, bmi, glucose level, insulin level, Blood pressure,etc, It also recommends the diet based on the different stages predicted.

Steps to execute:
✅ Step 1: Install Python
Make sure Python is installed (preferably Python 3.7+).
python --version
If not installed, download from python.org and check the Add to PATH option during install.

✅ Step 2: Clone the Project from GitHub
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
Replace with your actual GitHub repo link.

✅ Step 3: Create a Virtual Environment (Recommended)
python -m venv venv
Activate it:
On Windows:
venv\Scripts\activate
On Linux/macOS:
source venv/bin/activate

✅ Step 4: Install Dependencies
If your project has a requirements.txt file:
pip install -r requirements.txt
If not, install Flask manually:
pip install flask

✅ Step 5: Set Flask App and Run It
If your main app file is app.py, run:
On Linux/macOS:
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
On Windows CMD:
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
On PowerShell:
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
flask run

✅ Step 6: Open in Browser
After running, it will say:
Running on http://127.0.0.1:5000/
