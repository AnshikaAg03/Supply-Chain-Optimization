# Flask Supply Chain Optimization Project

## Project Overview
This Flask-based web application streamlines supply chain optimization by integrating *CO2 emissions-based taxation logic (by EU Standards)*. It offers demand simulation, country-wise analysis, and ESG-focused optimization tools to support decision-making for sustainability and efficiency.

---

## Features

### *Basic Optimization*
- *Upload and Preview Datasets*: Seamlessly upload and validate datasets for analysis.
- *Summarization*: 
   - Interactive globe visualization for data insights.
   - Paths with condensed information of the route
   - Uploaded Data Summarized in a visual manner 
- *Demand Simulation*: 
   - Model various demand scenarios based on uploaded data. 
   - Provides supply chain optimization for each of these Senarios.
- *Country-Wise Metrics*: View supply and demand breakdowns by country.

### *Advanced Analysis*
- *CO2 Taxation Insights*:
  - Summarize EU-specific CO2 tax impacts.
  - Integrate CO2 emissions data into optimization models.
- *ESG-Driven Optimization*:
  - Retrieve and analyze ESG scores for entities.
  - Perform ESG-focused optimization to align with sustainability goals.
- *Optimization with CO2 Scoring*
   - Incorporates CO2 data for actionable tax calculations.
   - Aligns with EU standards to encourage reduced emissions.

###*Detailed Results Display*
   - Comprehensive visualization of decision variables and optimization outcomes.
   - Separate sections for *'Optimized'* and *'Non-Optimized'* wrt CO2 emissions.

### *Enhanced Front-End*
- Intuitive and visually appealing interface.
- Tabs for uploading data, displaying results, and advanced analytics.

### *Simplified Code Management*
- Modularized codebase for easy maintenance.
- Proper comments to make the code easy to understand

---

## Technologies Used

- *Backend*: Flask  
- *Frontend*: HTML, CSS, JavaScript 
- *Optimization Tools Developed*: CO2 Tax rate calculator, ESG score Calculator , Supply chain visualizer , Demand Simulator.
- *Data Processing*: Pandas, NumPy 

---

## Installation Guide

1. *Clone the Repository*:
   bash
   git clone https://github.com/AnshikaAg03/Supply-Chain-Optimization/tree/main
   cd https://github.com/AnshikaAg03/Supply-Chain-Optimization/tree/main/flask-server/api
   

2. *Set Up Virtual Environment*:
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   

3. *Install Dependencies*:
   bash
   pip install -r requirements.txt
   

4. *Run the Application*:
   bash
   flask run
   
   The app will be accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## How to Use

1. *Upload Data*:
   - Use the *Upload File* tab to import your dataset.
   - Preview the uploaded data for correctness.

2. *Analyze Data*:
   - View pyplot glob for an visual breakdown of the upoaded data in the *Summarization* section.
   - Summarize CO2 taxes and explore demand simulations.

3. *Optimize Supply Chain*:
   - Click *Optimize* to calculate actionable results based on CO2 and ESG scores or just normal supply chain optimization if required so.

4. *View Results*:
   - Navigate to the respective tabs to explore optimized outputs based on CO2, ESG & profit basis.

---

## File Structure

flask-server/api/       
├── static             # Static files (CSS, JS, images)       
├── templates/         # HTML templates for the front-end
├── uploads/           # Temporary storage for user uploads
├── venv/              # Virtual environment files
├── app.py             # Main Flask application


---

## Customization

- *Modify ESG or Tax Logic*: Edit thresholds or calculations in app.py.
- *Enhance Front-End*: Update the design by modifying files in the templates/ directories.
- *Add New Features*: Extend the functionality in app.py as needed.

---

## Acknowledgments
- [Flask Documentation](https://flask.palletsprojects.com/)
- [EU CO2 Tax Guidelines](https://europa.eu/)

---
