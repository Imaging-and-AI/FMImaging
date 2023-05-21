
import wandb

# set up the sweep configuration
# first stage
sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'best_val_acc',
      'goal': 'maximize'
    },
    'parameters': {
        
        'num_epochs': {
            'values': [50, 100]
        },
        
        'batch_size': {
            'values': [32]
        },
        
        'global_lr': {
            'values': [1e-4, 2e-4]
        },
        
        'window_size': {
            'values': [[8, 8]]
        },
        
        'patch_size': {
            'values': [[4, 4], [2, 2]]
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
        
        'cell_type': {
            'values': ['sequential', 'parallel']
        },
        
        'n_head': {
            'values': [16, 32]
        },
        
        'mixer_type': {
            'values': ['conv', 'lin']
        },
        
        'normalize_Q_K': {
            'values': [True, False]
        },
        
        'cosine_att': {
            'values': [1, 0]
        },
        
        'att_with_relative_postion_bias': {
            'values': [1, 0]
        },
             
        'scale_ratio_in_mixer': {
            'values': [1.0, 2.0, 4.0]
        },
        
        'num_resolution_levels': {
            'values': [3, 2]
        },
        
        'C': {
            'values': [64, 128]
        },
        
        'block_str': {
            'values': [
                        ["T1L1G1", "T1L1G1", "T1L1G1"],
                        ["T1T1T1", "T1T1T1", "T1T1T1"],
                        ["T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"], 
                        ["T1L1G1T1L1G1", "T1L1G1T1L1G1", "T1L1G1T1L1G1"]
                    ]
        },
        
        'height': {
            'values': [[32, 64], [48, 96], [32], [64]]
        },
        
        'width': {
            'values': [[32, 64], [48, 96], [32], [64]]
        },
        
        'train_files': {
            'values': [
                [ ["train_3D_3T_retro_cine_2018.h5"], ["2dt"]], 
                [ ["train_3D_3T_retro_cine_2018.h5", "train_3D_3T_retro_cine_2019.h5"], ["2dt", "3d"] ],
                [ ["train_3D_3T_retro_cine_2018.h5", "train_3D_3T_retro_cine_2019.h5", "train_3D_3T_retro_cine_2020.h5"], ["2dt", "3d", "2d"] ]  
            ]            
        }
    }
}

def main():
    sweep_id = wandb.sweep(sweep_config, project="mri")
    
if __name__ == '__main__':
    main()