#!/bin/bash
python3 trajectory_handle_2014/raw_data_handle/prehandle.py > logs/1.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle2.py > logs/2.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle3.py > logs/3.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle4.py > logs/4.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle5.py > logs/5.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle6.py > logs/6.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle7.py > logs/7.log
