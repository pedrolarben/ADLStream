#!/bin/bash  
python ./scripts/benchmark_summary.py
git add .  
git commit -m $desc "test automatic commit" 
git push origin master
