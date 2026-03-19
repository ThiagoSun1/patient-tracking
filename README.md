# Patient Tracking

A Jetson Orin Nano-based patient tracking system **uses a YOLO model to classify behavior and have an AI supervisor to talk to the patient** .

---

## 🕹️😐 Overview

This project allows you to **keep track of patient statuses just by sitting in your office**.  

---

## 🧠 Features

- 🎯 Uses YOLO to track a patient and they're behavior
- 🎮 Uses a live camera stream
- ⚙️ Runs entirely on a Jetson Orin Nano
- 🐍 Written in pure Python with no external apps or software needed

---

## 📦 Hardware Requirements

- Jetson Orin Nano
  
---

## 🧰 Software Requirements

Before installing anything, make sure you're in a virtual enviroment:

```bash

# clone the repository
git clone https://github.com/ThiagoSun1/patient-tracking

# install package to create virtual enviroment
sudo apt install virtualenv

# make virtual enviroment
virtualenv venv

# activate virtual enviroment
source ~/venv/bin/activate

```

- Python 3.10 (pre-installed on Jetson Orin Nano in virtual envirments)
- The following Python libraries:

```bash
sudo apt update
sudo apt install python3-pip

# install necessary dependencies
pip install -r requirements.txt --force-reinstall

# run the tracker
cd ~/patient-tracking
python3 ollama+hpc.py

```
