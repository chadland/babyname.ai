#!/bin/bash

source /home/env/bin/activate
cd /home/lambdapack
rm -rf /tmp/*
python /host/local_test.py