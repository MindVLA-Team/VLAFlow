import json
import os
from pathlib import Path

import numpy as np

from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.evaluation import maniskill2_evaluator
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient

if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,  # to get unnormalization stats
        policy_setup=args.policy_setup,
        port=args.port,
        host=args.host,
        action_scale=args.action_scale,
        cfg_scale=1.5,  # cfg from 1.5 to 7 also performs well
        action_output_dim=args.action_output_dim,
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    output = Path(args.logging_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w") as stream:
        json.dump(
            {
                "env": args.env_name,
                "episodes": len(success_arr),
                "successes": int(sum(success_arr)),
                "success_rate": float(np.mean(success_arr)),
            },
            stream,
            indent=2,
        )
