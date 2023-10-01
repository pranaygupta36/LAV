pwd_root=$(pwd)


cd $CONDA_PREFIX/envs/CR-LAV-env/
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

echo "export CARLA_ROOT=/home/harpadmin/plant/carla" >> ./etc/conda/activate.d/env_vars.sh
echo "export CARLA_SERVER=\${CARLA_ROOT}/CarlaUE4.sh" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:leaderboard" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:scenario_runner" >> ./etc/conda/activate.d/env_vars.sh


echo "export LAV_ROOT=/home/harpadmin/vehicle_importance_human_annotation/psiturk-harp-template/public_files/generating_counterfactual_trajectories/LAV" >> ./etc/conda/activate.d/env_vars.sh
echo "export LEADERBOARD_ROOT=\${LAV_ROOT}/leaderboard" >> ./etc/conda/activate.d/env_vars.sh
echo "export SCENARIO_RUNNER_ROOT=\${LAV_ROOT}/scenario_runner" >> ./etc/conda/activate.d/env_vars.sh
echo "export TEAM_AGENT=\${LAV_ROOT}/team_code_v2/lav_agent_cr.py" >> ./etc/conda/activate.d/env_vars.sh
echo "export TEAM_CONFIG=\${LAV_ROOT}/team_code_v2/config.yaml" >> ./etc/conda/activate.d/env_vars.sh
echo "export SCENARIOS=\${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json" >> ./etc/conda/activate.d/env_vars.sh
echo "export REPETITIONS=1" >> ./etc/conda/activate.d/env_vars.sh
echo "export CHECKPOINT_ENDPOINT=results.json" >> ./etc/conda/activate.d/env_vars.sh
echo "export DEBUG_CHALLENGE=0" >> ./etc/conda/activate.d/env_vars.sh
echo "export CHALLENGE_TRACK_CODENAME=SENSORS" >> ./etc/conda/activate.d/env_vars.sh


echo "unset LAV_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CARLA_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CARLA_SERVER" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset PYTHONPATH" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset LEADERBOARD_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset SCENARIO_RUNNER_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset TEAM_AGENT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset TEAM_CONFIG" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset SCENARIOS" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset REPETITIONS" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CHECKPOINT_ENDPOINT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset DEBUG_CHALLENGE" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CHALLENGE_TRACK_CODENAME" >> ./etc/conda/deactivate.d/env_vars.sh



cd $pwd_root

