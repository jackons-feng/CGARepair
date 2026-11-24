#!/usr/bin/env python3
"""Convenience task entrypoint; callable from cfrepair_<task>/optimizer_study_20261031/."""
from pathlib import Path
import os
import subprocess
import sys

if __name__=='__main__':
    task_dir=Path(__file__).resolve().parent.parent
    project=task_dir.parent
    tasks={'adv':'adv','backdoor':'backdoor',
           'safety':'safety','fairness':'fairness'}
    task=tasks.get(task_dir.name)
    if task is None:raise SystemExit('Unexpected task directory: '+str(task_dir))
    runner=project/'optimizer_study_20261031'/'run_study.py'
    cmd=[sys.executable,str(runner),'--project-root',str(project),'--task',task]
    cmd+=sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))
