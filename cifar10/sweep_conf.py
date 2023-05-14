
import wandb

# set up the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'val_acc_1',
      'goal': 'maximize'
    },
    'parameters': {
        
        'num_epochs': {
            'values': [150, 100, 200]
        },
        
        'batch_size': {
            'values': [128]
        },
        
        'global_lr': {
            'values': [1e-4, 2e-4]
        },
        
        'weight_decay': {
            'values': [0.0, 0.1, 1.0]
        },
        
        'scheduler_type': {
            'values': ["ReduceLROnPlateau", "OneCycleLR"]
        },
        
        'use_amp': {
            'values': [True, False]
        },
        
        'a_type': {
            'values': ['conv', 'lin']
        },
        
        'n_head': {
            'values': [8, 12]
        },
        
        'scale_ratio_in_mixer': {
            'values': [1.0, 2.0, 4.0]
        },
        
        'num_resolution_levels': {
            'values': [3, 2]
        },
        
        'C': {
            'values': [32, 64, 128]
        },
        
        'block_str': {
            'values': [["T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"], 
                       ["T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"], 
                       ["T1L1G1", "T1L1G1", "T1L1G1"],
                       ["T1T1T1", "T1T1T1", "T1T1T1"]
                       ]
        }
    }
}

def main():
    sweep_id = wandb.sweep(sweep_config, project="cifar")
    
if __name__ == '__main__':
    main()