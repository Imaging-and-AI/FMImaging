"""
Base class to support run ddp experiments for projects
"""

import argparse
import itertools
import subprocess
import os
import shutil

class run_ddp_base(object):
    
    def __init__(self, project, script_to_run) -> None:
        super().__init__()
        self.project = project
        self.script_to_run = script_to_run
        self.cmd = []
        
    def set_up_torchrun(self, config):
        self.cmd = ["torchrun"]

        self.cmd.extend(["--nproc_per_node", f"{config.nproc_per_node}", "--max_restarts", "6"])

        if config.standalone:
            self.cmd.extend(["--standalone"])
        else:
            self.cmd.extend(["--nnodes", config.nnodes, 
                        "--node_rank", f"{config.node_rank}", 
                        "--rdzv_id", f"{config.rdzv_id}", 
                        "--rdzv_backend", f"{config.rdzv_backend}", 
                        "--rdzv_endpoint", f"{config.rdzv_endpoint}"])

        self.cmd.extend([self.script_to_run])

    
    def set_up_run_path(self, config):
        if "FMIMAGING_PROJECT_BASE" in os.environ:
            project_base_dir = os.environ['FMIMAGING_PROJECT_BASE']
        else:
            project_base_dir = '/export/Lab-Xue/projects'

        # unchanging paths
        
        ckp_path = os.path.join(project_base_dir, config.project, "checkpoints")
        
        if config.load_path is None:
            if config.clean_checkpoints:
                print(f"--> clean {ckp_path}")
                shutil.rmtree(ckp_path, ignore_errors=True)
                os.mkdir(ckp_path)
                
        data_root = config.data_root if config.data_root is not None else os.path.join(project_base_dir, config.project, "data")
            
        self.cmd.extend([
            "--data_root", data_root,
            "--check_path", ckp_path,
            "--model_path", os.path.join(project_base_dir, config.project, "models"),
            "--log_path", os.path.join(project_base_dir, config.project, "logs"),
            "--results_path", os.path.join(project_base_dir, config.project, "results"),            
        ])

        if config.with_timer:
            self.cmd.extend(["--with_timer"])


    def create_cmd_run(self, cmd_run, config, 
                        bk='hrnet', 
                        a_type='conv', 
                        cell_type='sequential', 
                        norm_mode='batch2d', 
                        block_dense_connection=1, 
                        c=32, 
                        q_k_norm=True, 
                        cosine_att=1, 
                        att_with_relative_postion_bias=1, 
                        bs=['T1G1L1', 'T1G1L1', 'T1G1L1', 'T1G1L1'],
                        larger_mixer_kernel=True,
                        mixer_type="conv",
                        shuffle_in_window=0,
                        scale_ratio_in_mixer=2.0,
                        load_path=None
                        ):
        
        run_str = f"{a_type}-{cell_type}-{norm_mode}-C-{c}-MIXER-{mixer_type}-{larger_mixer_kernel}-{int(scale_ratio_in_mixer)}-BLOCK_DENSE-{block_dense_connection}-QKNORM-{q_k_norm}-CONSINE_ATT-{cosine_att}-shuffle_in_window-{shuffle_in_window}-att_with_relative_postion_bias-{att_with_relative_postion_bias}-BLOCK_STR-{'_'.join(bs)}"
                                            
        cmd_run.extend([
            "--run_name", f"{config.project}-{bk.upper()}-{run_str}",
            "--run_notes", f"{config.project}-{bk.upper()}-{run_str}",
            "--backbone", f"{bk}",
            "--a_type", f"{a_type}",
            "--cell_type", f"{cell_type}",
            "--cosine_att", f"{cosine_att}",
            "--att_with_relative_postion_bias", f"{att_with_relative_postion_bias}",
            "--backbone_hrnet.C", f"{c}",
            "--backbone_unet.C", f"{c}",
            "--backbone_LLM.C", f"{c}",
            "--block_dense_connection", f"{block_dense_connection}",
            "--norm_mode", f"{norm_mode}",
            "--mixer_type", f"{mixer_type}",
            "--shuffle_in_window", f"{shuffle_in_window}",
            "--scale_ratio_in_mixer", f"{scale_ratio_in_mixer}"
        ])
        
        if larger_mixer_kernel:
            cmd_run.extend(["--mixer_kernel_size", "5", "--mixer_padding", "2", "--mixer_stride", "1"])
        else:
            cmd_run.extend(["--mixer_kernel_size", "3", "--mixer_padding", "1", "--mixer_stride", "1"])

        if q_k_norm:
            cmd_run.extend(["--normalize_Q_K"])
            
        cmd_run.extend([f"--backbone_{bk}.block_str", *bs])
        
        if load_path is not None:
            cmd_run.extend(["--load_path", load_path])
            
        print(f"Running command:\n{' '.join(cmd_run)}")

        return cmd_run

    def set_up_constants(self, config):    
        self.cmd.extend([       
        "--summary_depth", "6",
        "--save_cycle", "200",        
        "--device", "cuda",
        "--ddp", 
        "--project", self.project,
                               
        # hrnet
        "--backbone_hrnet.use_interpolation", "1",
        
        # unet            
        "--backbone_unet.use_unet_attention", "1",
        "--backbone_unet.use_interpolation", "1",
        "--backbone_unet.with_conv", "1",
        
        # LLMs
        "--backbone_LLM.add_skip_connections", "1"                        
        ])

    def set_up_variables(self, config):
        
        vars = dict()
        
        vars['backbone'] = ['hrnet']
        vars['cell_types'] = ["sequential"]
        vars['Q_K_norm'] = [True]
        vars['cosine_atts'] = ["1"]
        vars['att_with_relative_postion_biases'] = ["1"]
        vars['a_types'] = ["conv"]

        vars['larger_mixer_kernels'] = [False]
        vars['mixer_types'] = ["conv"]
        vars['shuffle_in_windows'] = ["0"]
        vars['block_dense_connections'] = ["1"]
        vars['norm_modes'] = ["batch2d"]
        vars['C'] = [64]
        vars['scale_ratio_in_mixers'] = [4.0]

        vars['block_strs'] = [
                        [["T1L1G1", "T1L1G1", "T1L1G1"], ["T1T1T1", "T1T1T1", "T1T1T1"] ]
                    ]

        return vars
    
    def run_vars(self, config, vars):
        
        for k, bk in enumerate(vars['backbone']):    
                block_str = vars['block_strs'][k]
                
                for bs in block_str:
                    for a_type, cell_type in itertools.product(vars['a_types'], vars['cell_types']):
                        for q_k_norm in vars['Q_K_norm']:
                            for cosine_att in vars['cosine_atts']:  
                                for att_with_relative_postion_bias in vars['att_with_relative_postion_biases']:
                                    for c in vars['C']:
                                        for block_dense_connection in vars['block_dense_connections']:
                                            for norm_mode in vars['norm_modes']:
                                                for larger_mixer_kernel in vars['larger_mixer_kernels']:
                                                    for shuffle_in_window in vars['shuffle_in_windows']:
                                                        for mixer_type in vars['mixer_types']:
                                                            for scale_ratio_in_mixer in vars['scale_ratio_in_mixers']:
                                                                
                                                                # -------------------------------------------------------------
                                                                cmd_run = self.create_cmd_run(cmd_run=self.cmd.copy(), 
                                                                                config=config,
                                                                                bk=bk, 
                                                                                a_type=a_type, 
                                                                                cell_type=cell_type,
                                                                                norm_mode=norm_mode, 
                                                                                block_dense_connection=block_dense_connection,
                                                                                c=c,
                                                                                q_k_norm=q_k_norm, 
                                                                                cosine_att=cosine_att, 
                                                                                att_with_relative_postion_bias=att_with_relative_postion_bias, 
                                                                                bs=bs,
                                                                                larger_mixer_kernel=larger_mixer_kernel,
                                                                                mixer_type=mixer_type,
                                                                                shuffle_in_window=shuffle_in_window,
                                                                                scale_ratio_in_mixer=scale_ratio_in_mixer,
                                                                                load_path=config.load_path)
                                                                
                                                                print("---" * 20)
                                                                print(cmd_run)
                                                                print("---" * 20)
                                                                subprocess.run(cmd_run)
                                                            
    def arg_parser(self):
        """
        @args:
            - No args
        @rets:
            - parser (ArgumentParser): the argparse for torchrun of mri
        """
        parser = argparse.ArgumentParser(prog=self.project)   
        
        parser.add_argument("--data_root", type=str, default=None, help="data folder; if None, use the project folder")
        
        parser.add_argument("--standalone", action="store_true", help='whether to run in the standalone mode')
        parser.add_argument("--nproc_per_node", type=int, default=2, help="number of processes per node")
        parser.add_argument("--nnodes", type=str, default="1", help="number of nodes")
        parser.add_argument("--node_rank", type=int, default=0, help="current node rank")
        parser.add_argument("--rdzv_id", type=int, default=100, help="run id")
        parser.add_argument("--rdzv_backend", type=str, default="c10d", help="backend of torchrun")
        parser.add_argument("--rdzv_endpoint", type=str, default="localhost:9001", help="master node endpoint")
        parser.add_argument("--load_path", type=str, default=None, help="check point file to load if provided")
        parser.add_argument("--clean_checkpoints", action="store_true", help='whether to delete previous check point files')
        parser.add_argument("--with_timer", action="store_true", help='whether to train with timing')
        
        args = parser.parse_args()
        
        return args

    def run(self):
        config = self.arg_parser()
        config.project = self.project
        self.set_up_torchrun(config)
        self.set_up_run_path(config)
        self.set_up_constants(config)
        vars = self.set_up_variables(config)
        self.run_vars(config, vars)

# -------------------------------------------------------------

def main():
    
    ddp_run = run_ddp_base(proj_info="stcnnt", script_to_run='./cifar/main_cifar.py')
    ddp_run.run()
         
# -------------------------------------------------------------

if __name__=="__main__":
    main()