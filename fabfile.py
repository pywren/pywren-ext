from fabric.api import local, env, run, put, cd, task, sudo, get, settings, warn_only, lcd
from fabric.contrib import project
import boto3
import cloudpickle
import json
import base64
from six.moves import cPickle as pickle
from pywren.wrenconfig import * 
import pywren
import time

env.roledefs['m'] = ['jonas@c65']


@task
def deploy():
        local('git ls-tree --full-tree --name-only -r HEAD > .git-files-list')
    
        project.rsync_project("/data/jonas/pywren-ext/", local_dir="./",
                              exclude=['*.npy', "*.ipynb", 'data', "*.mp4", 
                                       "*.pdf", "*.png"],
                              extra_opts='--files-from=.git-files-list')

        # # copy the notebooks from remote to local

        project.rsync_project("/data/jonas/pywren-ext/", local_dir="./",
                              extra_opts="--include '*.ipynb' --include '*.pdf' --include '*.png'  --include='*/' --exclude='*' ", 
                              upload=False)
        
