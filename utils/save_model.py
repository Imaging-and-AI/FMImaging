"""
shared file to save general model
provides the following functions:
    - generate_model_file_name
    - save_final_model
"""
import os
import json
import torch
import logging

# -------------------------------------------------------------------------------------------------
# basic file name generator using the config

def generate_model_file_name(config):
    """
    generate file name from the config
    @args:
        - config (Namespace): runtime namespace for setup
    @args (from config):
        - run_name (str): the name of the run
        - data (str): the datetime when the run was started
    @rets:
        - model_file_name (str): the generated file name
    """
    run_name = config.run_name
    date = config.date

    return f"{run_name}_{date}"

# -------------------------------------------------------------------------------------------------
# function to save generic model

def save_final_model(model, config, best_model_wts):
    """
    save the model as ".pt+.json" and ".pts", and save again after loading the best_model_wts
    @args:
        - model (torch model): model to be saved
        - config (Namespace): runtime namespace for setup
        - best_model_wts (model.state_dict): the best model weight during the run
    @rets:
        - last_model_name, best_model_name (str):
            the path the last and the best saved models respectively
    """
    c = config # shortening due to numerous uses

    model_input = torch.randn(1, c.time, c.C_in, c.height[0], c.width[0], requires_grad=True)
    model_input = model_input.to('cpu')
    model.to('cpu')

    model_file_name = os.path.join(c.model_path, generate_model_file_name(config))

    def save_model_instance(model, name):
        # save an instance of the model using the given name
        logging.info(f"Saving model weights at: {name}.pt")
        torch.save(model.state_dict(), f"{name}.pt")

        logging.info(f"Saving torchscript model at: {name}.pts")
        model_scripted = torch.jit.trace(model, model_input, strict=False)
        model_scripted.save(f"{name}.pts")

        logging.info(f"Saving config at {name}.json")
        with open(f"{name}.json", "w") as file:
            try:
                config_dict = dict(config)
            except TypeError:
                config_dict = vars(config)
            json.dump(config_dict, file)

    last_model_name = f"{model_file_name}_last"
    save_model_instance(model, name=last_model_name)
    best_model_name = f"{model_file_name}_best"
    model.load_state_dict(best_model_wts)
    save_model_instance(model, name=best_model_name)

    logging.info(f"All saving complete")

    return last_model_name, best_model_name
