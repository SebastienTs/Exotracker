@echo off
set USER_HOME=C:\Users\sebas
call %USER_HOME%\miniconda3\Scripts\activate exotracker
python main.py
call conda deactivate
