# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


from gymnasium.envs.registration import register


def register_highway_envs():
    """Import the envs module so that envs register themselves."""

    register(
        id='safe-highway-v0',
        entry_point='highway_env.envs:SafeHighwayEnv',
    )

    register(
        id='highway-fast-v0',
        entry_point='highway_env.envs:HighwayEnvFast',
    )

    # intersection_env.py
    register(
        id='intersection-v0',
        entry_point='highway_env.envs:IntersectionEnv',
    )

    register(
        id='intersection-v1',
        entry_point='highway_env.envs:ContinuousIntersectionEnv',
    )

    register(
        id='intersection-multi-agent-v0',
        entry_point='highway_env.envs:MultiAgentIntersectionEnv',
    )

    register(
        id='intersection-multi-agent-v1',
        entry_point='highway_env.envs:TupleMultiAgentIntersectionEnv',
    )

    # merge_env.py
    register(
        id='merge-v0',
        entry_point='highway_env.envs:MergeEnv',
    )

    register(
        id='merge-v1',
        entry_point='highway_env.envs:MergeEnvTest',
    )

    # roundabout.py
    register(
        id='roundabout-v0',
        entry_point='highway_env.envs:RoundaboutEnv',
    )


