Coimbra Coolectiva Mapping Tool
===============================

Overview
--------

This repository hosts the source code for an interactive web-based mapping and data visualization tool focused on the Coimbra region. Built with Python using the Streamlit framework, this tool leverages various data manipulation and visualization libraries to offer a robust analysis platform.

Folder and File Structure
-------------------------
- **Preloaded/** - Contains preloaded data files from [INE.pt](https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_doc_municipios)
- **_pycache_/** - Python cache files from local execution.
- **maps/** - Geographic data files.
- **tables/** - CSV files from [INE](https://www.ine.pt/xportal/xmain?xpgid=ine_main&xpid=INE).
- **coimbra_app.py** - Main application file for running the Streamlit web app.
- **locations.py** - Defines geographic structure and locations data.
- **preprocess.py** - Contains data preprocessing functions.
- **requirements.txt** - Required libraries for the project.
- **README.md** - The file you are currently reading.

Installation
------------

To set up and run the application locally:

1.  Clone the repository:

    `git clone https://github.com/your-repository/coimbra_app.git`

2.  Navigate to the project directory:

    `cd coimbra_app`

3.  Install dependencies:

    `pip install -r requirements.txt`

Usage
-----

Run the application using Streamlit:

`streamlit run coimbra_app.py`

This command will start the Streamlit server and open the application in your default web browser.

Features
--------

-   Interactive Maps: Visualize geographic and demographic data interactively on maps.
-   Data Analysis: Process and analyze data using integrated Python libraries.
-   Chart Generation: Dynamically generate charts based on user inputs and selections.
-   Data Customization: Users can upload their datasets for personalized analysis.

Contributing
------------

We welcome contributions to enhance the application's functionalities or documentation. Please fork the repository and submit a pull request with your changes. Major changes should be discussed via an issue first.

## Course Information

This project is part of the coursework for the University of Manchester's Data Science program, specifically for the course titled "Applying Data Science."
