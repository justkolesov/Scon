import argparse
import yaml
import os
import logging
import  sys
import torch
import numpy as np
import traceback
import shutil
import wandb

 
import util 
from compatibility.runners import CpatRunner
from baryproj.runners import BPRunner
from scones.runners import  GaussianRunner, SCONESRunner


def main():
    
    #============PARSE_BLOCK============#
    print(util.green("==============================="))
    print(util.green("=========== Scones ============"))
    print(util.green("==============================="))
    print(util.magenta(" main.py: Parsing command line..."))
    
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.load(file)
     
    new_config = util.dict2namespace(config)
    print(util.yellow(" main.py : Config is uploaded"))
    #============PARSE_BLOCK============#
    
    
    #===========MAKEDIRS=================#
    print(util.magenta(" main.py : makedirs ..."))
    if not os.path.exists(new_config.compatibility.logging.log_path):
        os.makedirs(new_config.compatibility.logging.log_path)
       
    if not os.path.exists(new_config.baryproj.logging.log_path):
        os.makedirs(new_config.baryproj.logging.log_path)
        
    if not os.path.exists(new_config.scones.logging.log_path):
        os.makedirs(new_config.scones.logging.log_path)
    print(util.magenta(" main.py : makedirs are ready!"))
    
    #===========MAKEDIRS=================#
    
    
   
    #================LOGGING============#
    print(util.green(" main.py : Set logging and Handlers "))
    print(util.green("..."))
    
    #setup logger
    logger = logging.getLogger()
    
    #create two handler for logging to sys.stdout and 
    # to file
    handler_stream = logging.StreamHandler()
    handler_cpat = logging.FileHandler(os.path.join(new_config.compatibility.logging.log_path, 'stdout.txt'))
    handler_bproj = logging.FileHandler(os.path.join(new_config.baryproj.logging.log_path, 'stdout.txt'))
    handler_scones = logging.FileHandler(os.path.join(new_config.scones.logging.log_path, 'stdout.txt'))
    
    # define formate of logging
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler_stream.setFormatter(formatter)
    handler_cpat.setFormatter(formatter)
    handler_bproj.setFormatter(formatter)
    handler_scones.setFormatter(formatter)
    
    # set logging level for stderr and files
    level = getattr(logging, new_config.compatibility.logging.verbose_stderr.upper(), None)
    handler_cpat.setLevel(level)
    
    level = getattr(logging, new_config.baryproj.logging.verbose_stderr.upper(), None)
    handler_bproj.setLevel(level)
    
    level = getattr(logging, new_config.scones.logging.verbose_stderr.upper(), None)
    handler_scones.setLevel(level)
    
    level = getattr(logging, new_config.logging.verbose_stderr.upper(), None)
    handler_stream.setLevel(level)
    
    # add handlers to logger
    logger.addHandler(handler_cpat)
    logger.addHandler(handler_bproj)
    logger.addHandler(handler_scones)
    logger.addHandler(handler_stream)
    
    # set level logging for logger
    level = getattr(logging, new_config.logging.verbose_logger.upper(), None)
    logger.setLevel(level)
    print(util.green(' main.py : Logger is ready!'))
    #================LOGGING==================#
    
    
    
    #=============RESUME_BLOCK==========#
    print(util.yellow(" main.py : resume for Cpat... "))
    if not new_config.compatibility.logging.resume_training:
        if os.path.exists(new_config.compatibility.logging.log_path):
            overwrite = False
            response = input("Folder already exists. Overwrite? (Y/N) ")
            if response.upper() == 'Y':
                overwrite = True

            if overwrite:
                shutil.rmtree(new_config.compatibility.logging.log_path)
                os.makedirs(new_config.compatibility.logging.log_path)
                
                
                #if os.path.exists(tb_path):
                #   shutil.rmtree(tb_path)
                
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(new_config.compatibility.logging.log_path)
            
    with open(os.path.join(new_config.compatibility.logging.log_path, 'config.yml'), 'w+') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    print(util.yellow(" main.py : resume for Cpat is ready "))
    #=============RESUME_BLOCK==========#
    
    
    
    #==============DEVICE_BLOCK===============#
    print(util.magenta("Preparing of devices..."))
    assert torch.cuda.is_available()
    device = torch.device("cuda:{}".format(new_config.device_gpus.ngpu[0]))
    new_config.device = device
    logging.info("Using device: {}".format(device))
    print(util.magenta(f" main.py : device is {device}"))
    #==============DEVICE_BLOCK===============#
    
    
    #=============SET_SEED====================#
    print(util.magenta("Set seed..."))
    torch.manual_seed(new_config.compatibility.training.seed)
    np.random.seed(new_config.compatibility.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(new_config.compatibility.training.seed)
    
    torch.backends.cudnn.benchmark = True
    logging.info(f"Using seed {new_config.compatibility.training.seed}" )
    print(util.magenta(" main.py : seed is ready"))
    #=============SET_SEED====================#
    
    
    
    wandb.init(project="scones_celeba",name=f"dim_{new_config.source.data.dim}_eps_{new_config.transport.coeff}")
    
    
    
    #============= CPAT =================#
    print(util.yellow(" Compatibility part ..."))
    try:
        runner = CpatRunner(new_config)
        runner.train()
         
    except:
        logging.error(traceback.format_exc())
    print(util.yellow("Compatibility is ready!"))
    #===================================#
    
    
    """
    #============== BProj ==============#
    print(util.green(" Barycentric projection part ..."))
    try:
        runner_bproj =  BPRunner(new_config)
        runner_bproj.train()
    except:
        logging.error(traceback.format_exc())
    print(util.green(" Barycentric projection is ready!"))
    #===================================#
    """
    
    
    #==============SCONEs===============#
    print(util.magenta("Scones  part..."))
    prefixes = ["gaussian","mixgauss"]
    if any(new_config.meta.problem_name.startswith(x) for x in prefixes ):
        runner_scones =  GaussianRunner(new_config)
        
    elif new_config.meta.problem_name.startswith("CELEBA"):
        runner_scones = SCONESRunner(new_config)
        
    runner_scones.sample()
    #===================================#
    

if __name__ == '__main__':
    main()