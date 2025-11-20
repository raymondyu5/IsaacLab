

./isaaclab.sh -p source/standalone/workflows/sysID/ASID/random_agent.py --task Isaac-Lift-DeformCube-Franka-v0 --env_config source/config/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction] --enable_camera --parmas_range [10] --headles

pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/sysID/ASID/random_agent.py --task Isaac-Lift-DeformCube-Franka-v0 --env_config source/config/rabbit_env_random.yaml --num_envs=8 --random_params [youngs_modulus] --enable_camera --parmas_range [10] --headles

pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/sysID/ASID/random_agent.py --task Isaac-Lift-DeformCube-Franka-v0 --env_config source/config/rabbit_env_random.yaml --num_envs=8 --random_params [elasticity_damping] --enable_camera --parmas_range [10] --headles

pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/sysID/ASID/random_agent.py --task Isaac-Lift-DeformCube-Franka-v0 --env_config source/config/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction,youngs_modulus] --enable_camera --parmas_range [8,8] --headles

pkill -f "python"
sleep 10
./isaaclab.sh -p source/standalone/workflows/sysID/ASID/random_agent.py --task Isaac-Lift-DeformCube-Franka-v0 --env_config source/config/rabbit_env_random.yaml --num_envs=8 --random_params [dynamic_friction,elasticity_damping] --enable_camera --parmas_range [8,8] --headles