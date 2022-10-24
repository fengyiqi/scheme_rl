def data_path(name: str, time: str):
    if name.lower() == "implosion":
        files = {
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
        return files
    elif name.lower() == "config3":
        files = {
            64: [
                f"/media/yiqi/Elements/RL/baseline/config3_64_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_64_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_64_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/config3/RiemannConfig3Env/runtime_data/config3_64_{time}/domain/data_{time}000.h5",
            ],
            128: [
                f"/media/yiqi/Elements/RL/baseline/config3_128_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_128_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/config3_128_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/config3/RiemannConfig3HighRes128Env/runtime_data/config3_128_{time}/domain/data_{time}000.h5",
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
        return files
    elif name.lower() == "shear":
        files = { 
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
        return files
    elif name.lower() == "rti":
        files = { 
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
            256: [
                f"/media/yiqi/Elements/RL/baseline/rti_256_weno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_256_teno5/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/baseline/rti_256_teno5lin/domain/data_{time}*.h5",
                f"/media/yiqi/Elements/RL/August/rti/RTIHighRes256Env/runtime_data/rti_256_{time}/domain/data_{time}000.h5",
            ]
        }
        return files
    else:
        raise "No such configuration!"
