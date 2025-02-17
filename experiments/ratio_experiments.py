"""
ratio_experiments.py

Script that automatically runs the classification training at multiple
real-to-synthetic ratios (dermatofibroma only), then evaluates each model.
"""

import subprocess

# Example ratios: 0, 0.5, 1.0, 1.5, 2.0
RATIOS = [0, 0.5, 1.0, 1.5, 2.0]


def run_experiments():
    for r in RATIOS:
        checkpoint_name = f"efficientnet_v2_synth_{r}.pth"

        print(f"\n=== Training with synthetic_ratio={r} ===")
        # You could run classification.py directly and pass arguments
        subprocess.run(
            [
                "python",
                "src/classification.py",
                str(r),  # if classification.py was set up to parse from sys.argv
            ]
        )

        # or call the function via some other method
        # for now, let's assume you only run classification.py's main with ratio=0 by default
        # so you might need to edit classification.py to parse arguments from the command line

        print(f"\n=== Evaluating ratio={r} ===")
        subprocess.run(["python", "src/evaluation.py", checkpoint_name])


if __name__ == "__main__":
    run_experiments()
