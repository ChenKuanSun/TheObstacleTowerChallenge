from obstacle_tower_env import ObstacleTowerEnv
import sys,os,time
import argparse
base_dir ='./save/'

def create_otc_environment(env):
    print("Create env")
    env = unity_lib.OTCPreprocessing(env)
    return env

def create_agent(sess, environment,
                 agent_name=None,
                 summary_writer=None,
                 debug_mode=False):
    return rainbow_agent.RainbowAgent(sess,
                                      num_actions=8,
                                      summary_writer=summary_writer)
def run_episode(runner):
    _, total_reward = runner._run_one_episode()
    return total_reward

def run_evaluation(runner, env):
    while not env.done_grading():
        run_episode(runner)
        runner._environment.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()
    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, retro=False, timeout_wait=304)
    
    if env.is_grading():
        from obstacle_tower_env import ObstacleTowerEnv
        from dopamine.agents.rainbow import rainbow_agent
        from dopamine.discrete_domains import run_experiment
        from keepitpossible.common import unity_lib
        runner = run_experiment.TrainRunner(base_dir,
                                            create_agent,
                                            create_environment_fn=create_otc_environment(env))
        episode_reward = run_evaluation(runner, env)
        print(episode_reward)
    else:
        from obstacle_tower_env import ObstacleTowerEnv
        from dopamine.agents.rainbow import rainbow_agent
        from dopamine.discrete_domains import run_experiment
        from keepitpossible.common import unity_lib
        runner = run_experiment.TrainRunner(base_dir,
                                            create_agent,
                                            create_environment_fn=create_otc_environment(env))
        while True:
            episode_reward = run_episode(runner)
            print(episode_reward)
            runner._environment.reset()
