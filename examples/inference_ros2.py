import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


config = _config.get_config("pi0_aloha_pen_uncap")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = aloha_policy.make_aloha_example()
result = policy.infer(example)

print("Actions shape:", result["actions"].shape)
