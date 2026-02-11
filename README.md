# My submission for Bosch's Applied CV Coding Assignment
[![Project Validation](https://github.com/plsgivecheesecake/bosch-applied-cv-assignment/actions/workflows/ci.yml/badge.svg)](https://github.com/plsgivecheesecake/bosch-applied-cv-assignment/actions/workflows/ci.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## Setup
Please make sure to do the following
 - Set environment variable called BDD_DATASET_PATH with the directory path where the **EXTRACTED** contents of the assignment_data_bdd.zip file are located on your system. This can be done in one of two ways. You can either set this as an OS variable, or you can use the .env file provided in this repository. If on Windows, please make sure to escape the backslashes or just use a UNIX-style path.
 - Install Docker and Docker Compose

Once the above two steps have been completed, running the project is as simple as running 
``` docker compose up --build ``` from the root of the repository.

When the app is up and running, please navigate to http://localhost:8501/ to access the UI. You will be greeted by a page that looks like this
![Process Dataset](readme_images/process_dataset.jpg)

Please make sure to click the "Process Dataset" button before proceeding to the next pages.
For ease of reading, I have divided task 1 into 5 pages. last two pages are for task 2 and task 3.

## Task 1 : Dataset Analysis
