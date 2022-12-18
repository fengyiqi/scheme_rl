def data_path(name: str, time: str):
    """
    Return corresponding data file path
    """
    if name.lower() == "implosion":
        return {
            64: [
                f"/media/yiqi/Elements/RL/baseline/implosion_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/implosion_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/implosion_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/implosion/ImplosionEnv/runtime_data/implosion_64_{time}/domain/data_{time}000.h5",
            ],
            128: [
                f"/media/yiqi/Elements/RL/baseline/implosion_128_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/implosion_128_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/implosion_128_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/implosion/ImplosionHighRes128Env/runtime_data/implosion_128_{time}/domain/data_{time}000.h5",
            ],
            256: [
                f"/media/yiqi/Elements/RL/baseline/implosion_256_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/implosion_256_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/implosion_256_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/implosion/ImplosionHighRes256Env/runtime_data/implosion_256_{time}/domain/data_{time}000.h5",
            ]
        }
    elif name.lower() == "config3":
        return {
            64: [
                f"/media/yiqi/Elements/RL/baseline/config3_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/config3/RiemannConfig3Env/runtime_data/config3_64_{time}/domain/data_{time}000.h5",
            ],
            256: [
                f"/media/yiqi/Elements/RL/baseline/config3_256_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_256_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_256_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/config3/RiemannConfig3HighRes256Env/runtime_data/config3_256_{time}/domain/data_{time}000.h5",
            ],
            512: [
                f"/media/yiqi/Elements/RL/baseline/config3_512_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_512_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_512_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/config3/RiemannConfig3HighRes512Env/runtime_data/config3_512_{time}/domain/data_{time}000.h5",
            ]
        }
    elif name.lower() == "shear":
        return { 
            64: [
                f"/media/yiqi/Elements/RL/baseline/shear_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/shear/FreeShearEnv/runtime_data/shear_64_{time}/domain/data_{time}000.h5",
            ],
            128: [
                f"/media/yiqi/Elements/RL/baseline/shear_128_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_128_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_128_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/shear/FreeShearHighRes128Env/runtime_data/shear_128_{time}/domain/data_{time}000.h5",
            ],
            256: [
                f"/media/yiqi/Elements/RL/baseline/shear_256_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_256_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_256_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/shear/FreeShearHighRes256Env/runtime_data/shear_256_{time}/domain/data_{time}000.h5",
            ]
        }
    elif name.lower() == "shear_thin":
        return { 
            64: [
                f"/media/yiqi/Elements/RL/baseline/shear_thin_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_thin_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_thin_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/shear_thin/FreeShearThinEnv/runtime_data/shear_thin_64_{time}/domain/data_{time}000.h5",
            ],
            128: [
                f"/media/yiqi/Elements/RL/baseline/shear_thin_128_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_thin_128_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_thin_128_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/shear_thin/FreeShearThinHighRes128Env/runtime_data/shear_thin_128_{time}/domain/data_{time}000.h5",
            ],
            256: [
                f"/media/yiqi/Elements/RL/baseline/shear_thin_256_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_thin_256_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/shear_thin_256_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/shear_thin/FreeShearThinHighRes256Env/runtime_data/shear_thin_256_{time}/domain/data_{time}000.h5",
            ]
        }
    elif name.lower() == "rti":
        return { 
            32: [
                f"/media/yiqi/Elements/RL/baseline/rti_32_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_32_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_32_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/rti/RTIEnv/runtime_data/rti_32_{time}/domain/data_{time}000.h5",
            ],
            64: [
                f"/media/yiqi/Elements/RL/baseline/rti_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/rti/RTIHighRes64Env/runtime_data/rti_64_{time}/domain/data_{time}000.h5",
            ],
            128: [
                f"/media/yiqi/Elements/RL/baseline/rti_128_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_128_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_128_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/rti/RTIHighRes128Env/runtime_data/rti_128_{time}/domain/data_{time}000.h5",
            ],
        }
    elif name.lower() == "viscous_shock":
        return { 
            64: [
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/viscous_shock/ViscousShockTubeEnv/runtime_data/viscous_shock_64_{time}/domain/data_{time}000.h5",
            ],
            128: [
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_128_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_128_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_128_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/viscous_shock/ViscousShockTubeHighRes128Env/runtime_data/viscous_shock_128_{time}/domain/data_{time}000.h5",
            ],
            256: [
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_256_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_256_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_256_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/viscous_shock/ViscousShockTubeHighRes256Env/runtime_data/viscous_shock_256_{time}/domain/data_{time}000.h5",
            ],
            512: [
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_512_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_512_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/viscous_shock_512_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/viscous_shock/ViscousShockTubeHighRes512Env/runtime_data/viscous_shock_512_{time}/domain/data_{time}000.h5",
            ],
        }
    elif name.lower() == "moving_gresho":
        return { 
            32: [
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_32_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_32_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_32_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/moving_gresho/MovingGreshoEnv/runtime_data/moving_gresho_32_{time}/domain/data_{time}000.h5",
            ],
            64: [
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/moving_gresho/MovingGreshoHighRes64Env/runtime_data/moving_gresho_64_{time}/domain/data_{time}000.h5",
            ],
            128: [
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_128_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_128_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/moving_gresho_128_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/moving_gresho/MovingGreshoHighRes128Env/runtime_data/moving_gresho_128_{time}/domain/data_{time}000.h5",
            ],
        }
    else:
        raise "No such configuration!"


def envs_name(name: str):
    """
    Return corresponding environment name
    """
    if name.lower() == "implosion":
        return {
            64: "ImplosionEnv", 
            128: "ImplosionHighRes128Env", 
            256: "ImplosionHighRes256Env"
        }
    elif name.lower() == "config3":
        return {
            64: "RiemannConfig3Env", 
            128: "RiemannConfig3HighRes128Env", 
            256: "RiemannConfig3HighRes256Env", 
            512: "RiemannConfig3HighRes512Env"
        }
    elif name.lower() == "rti":
        return {
            32: "RTIEnv",
            64: "RTIHighRes64Env",
            128: "RTIHighRes128Env",
            256: "RTIHighRes256Env"
        }
    elif name.lower() == "shear":
        return {
            64: "FreeShearEnv", 
            128: "FreeShearHighRes128Env", 
            256: "FreeShearHighRes256Env"
        }
    elif name.lower() == "viscous_shock":
        return {
            64: "ViscousShockTubeEnv", 
            128: "ViscousShockTubeHighRes128Env", 
            256: "ViscousShockTubeHighRes256Env",
            512: "ViscousShockTubeHighRes512Env"
        }
    elif name.lower() == "moving_gresho":
        return {
            32: "MovingGreshoEnv", 
            64: "MovingGreshoHighRes64Env", 
            128: "MovingGreshoHighRes128Env",
        }
    elif name.lower() == "shear_thin":
        return {
            64: "FreeShearThinEnv", 
            128: "FreeShearThinHighRes128Env", 
            256: "FreeShearThinHighRes256Env",
        }
    else:
        raise "No such configuration!"

def time_config(name: str):
    """
    Return corresponding time configuration
    """
    if name.lower() == "implosion":
        return {
            "end_time": 2.5, 
            "timestep_size": 0.01, 
        }
    elif name.lower() == "config3":
        return {
            "end_time": 1.1, 
            "timestep_size": 0.01, 
        }
    elif name.lower() == "rti":
        return {
            "end_time": 2.0, 
            "timestep_size": 0.01, 
        }
    elif name.lower() == "shear":
        return {
            "end_time": 1.0, 
            "timestep_size": 0.01, 
        }
    elif name.lower() == "shear_thin":
        return {
            "end_time": 1.0, 
            "timestep_size": 0.01, 
        }
    elif name.lower() == "viscous_shock":
        return {
            "end_time": 1.0, 
            "timestep_size": 0.01, 
        }
    elif name.lower() == "moving_gresho":
        return {
            "end_time": 3.0, 
            "timestep_size": 0.01, 
        }
    else:
        raise "No such configuration!"

def shape_config(name: str):
    """
    Return corresponding shape configuration
    """
    if name.lower() == "implosion":
        return {
            64: (64, 64),
            128: (128, 128), 
            256: (256, 256)
        }
    elif name.lower() == "config3":
        return {
            64: (64, 64), 
            128: (128, 128), 
            256: (256, 256), 
            512: (512, 512)
        }
    elif name.lower() == "rti":
        return {
            32: (128, 32),
            64: (256, 64),
            128: (512, 128),
            256: (1024, 256)
        }
    elif name.lower() == "shear":
        return {
            64: (64, 64), 
            128: (128, 128), 
            256: (256, 256)
        }
    elif name.lower() == "shear_thin":
        return {
            64: (64, 64), 
            128: (128, 128), 
            256: (256, 256)
        }
    elif name.lower() == "viscous_shock":
        return {
            64: (32, 64), 
            128: (64, 128), 
            256: (128, 256),
            512: (256, 512)
        }
    elif name.lower() == "moving_gresho":
        return {
            32: (32, 128), 
            64: (64, 256), 
            128: (128, 512),
            # 512: (256, 512)
        }
    else:
        raise "No such configuration!"

def reward_files(name: str) -> list:
    if name.lower() == "implosion":
        return [
            "/media/yiqi/Elements/RL/August/implosion/ppo_models/progress_100.csv",
            "/media/yiqi/Elements/RL/August/implosion/ppo_models/progress_200.csv",
            "/media/yiqi/Elements/RL/August/implosion/ppo_models/progress_300.csv"
        ]
    elif name.lower() == "config3":
        return [
            "/media/yiqi/Elements/RL/August/config3/ppo_models/progress_100.csv",
            "/media/yiqi/Elements/RL/August/config3/ppo_models/progress_200.csv",
            "/media/yiqi/Elements/RL/August/config3/ppo_models/progress_300.csv"
        ]
    elif name.lower() == "shear":
        return [
            "/media/yiqi/Elements/RL/August/shear/ppo_models/progress_100.csv",
            "/media/yiqi/Elements/RL/August/shear/ppo_models/progress_200.csv",
            "/media/yiqi/Elements/RL/August/shear/ppo_models/progress_300.csv"
        ]

def deterministic_reward_files(name: str) -> list:
    if name.lower() == "implosion":
        return [
            "/media/yiqi/Elements/RL/August/implosion/ppo_models/implosion_100_rewards.csv",
            "/media/yiqi/Elements/RL/August/implosion/ppo_models/implosion_200_rewards.csv",
            "/media/yiqi/Elements/RL/August/implosion/ppo_models/implosion_300_rewards.csv"
        ]
    elif name.lower() == "config3":
        return [
            "/media/yiqi/Elements/RL/August/config3/ppo_models/config3_100_rewards.csv",
            "/media/yiqi/Elements/RL/August/config3/ppo_models/config3_200_rewards.csv",
            "/media/yiqi/Elements/RL/August/config3/ppo_models/config3_300_rewards.csv"
        ]
    elif name.lower() == "shear":
        return [
            "/media/yiqi/Elements/RL/August/shear/ppo_models/shear_100_rewards.csv",
            "/media/yiqi/Elements/RL/August/shear/ppo_models/shear_200_rewards.csv",
            "/media/yiqi/Elements/RL/August/shear/ppo_models/shear_300_rewards.csv"
        ]

def data_slices(name: str) -> slice:
    if name.lower() == "moving_gresho":
        return {
            32: slice(96, None),
            64: slice(192, None),
            128: slice(384, None)
        }
    else:
        return None