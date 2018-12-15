#!/bin/bash
python3 trajectory_handle_2014/raw_data_handle/prehandle.py > /root/logs/1.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle2.py > /root/logs/2.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle3.py > /root/logs/3.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle4.py > /root/logs/4.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle5.py > /root/logs/5.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle6.py > /root/logs/6.log & \
python3 trajectory_handle_2014/raw_data_handle/prehandle7.py > /root/logs/7.log
