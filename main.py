import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
# from openpi_client.runtime_ros2 import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import time
import os
print(os.getcwd())

# from examples.aloha_real import env as _env
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
    InterbotixRobotNode
)

from examples.aloha_real import env_ros2 as _env
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
import threading



@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25



class Runtime(InterbotixRobotNode):
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        metadata,
        ws_client_policy
    ) -> None:
        super().__init__('runtime_node')
        
        action_horizon = 25
        num_episodes = 1
        max_episode_steps = 1000

        self._environment = _env.AlohaRealEnvironment(node = self, reset_position=metadata.get("reset_pose"))
        self._agent =_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=action_horizon,
            )
        )

        subscribers=[]

        max_hz=50

        print("self._environment: ", self._environment )

        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = 1000

        self._in_episode = False
        self._episode_steps = 0

        timer_period = 1  # seconds
        # print("waiting")
        # time.sleep(4)
        self.timer = self.create_timer(timer_period, self.run)

        # self.timer_thread = threading.Thread(target=self.run, daemon=True)
        # self.timer_thread.start()

    def timer_loop(self):
        while rclpy.ok():
            self.get_logger().info('Timer callback started')
            time.sleep(5)  # Simulate long computation
            self.get_logger().info('Timer callback ended')
            time.sleep(2)  # Wait for next timer execution

    def run(self) -> None:

        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()
            # print("in run!!!!!!!!!!!!!!!!!!!!!!")
            # time.sleep(1)
            # print("end run!!!!!!!!!!!!!!!!!!!!!!")

        # # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()


    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """A single step of the runtime loop."""
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()





def main() -> None:
    rclpy.init()

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host="0.0.0.0",
        port=8000,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()

    # node = create_interbotix_global_node('aloha')
    node = Runtime( metadata , ws_client_policy)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

    


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, force=True)
    # tyro.cli(main)
    main()
