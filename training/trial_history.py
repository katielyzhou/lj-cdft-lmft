import os
import json
import matplotlib.pyplot as plt
from kerastuner.tuners import GridSearch as _GS
from kerastuner.engine.trial import Trial


class GridSearch(_GS):
    histories = {}

    def on_epoch_end(self, trial, model, epoch, logs=None):
        trial_id = trial.trial_id
        trial_history = self.histories.setdefault(trial_id, {})
        for metric, value in logs.items():
            trial_history.setdefault(metric, []).append(value)

    def on_trial_end(self, trial):
        super().on_trial_end(trial)

        self.plot_and_save_history(trial)

    def plot_and_save_history(self, trial: Trial):
        # this will save a plot of the trial training metrics by step in each train folder, it'll also save the metrics as a .json
        trial_id = trial.trial_id
        trial_history = self.histories[trial_id]

        # Create directory for trial if it doesn't exist
        trial_dir = self.get_trial_dir(trial_id)
        os.makedirs(trial_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))  # Create a larger figure

        for metric_name, metric_values in trial_history.items():
            plt.plot(metric_values, label=metric_name)  # Add label for legend

        plt.title("Metrics History")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()  # Add legend
        plt.savefig(os.path.join(trial_dir, "metrics.png"))
        plt.close()

        # Save history to JSON file
        history_file = os.path.join(trial_dir, "history.json")
        with open(history_file, "w") as f:
            json.dump(trial_history, f, indent=4)

    def plot_best_trials(self, num_models, wrap_columns=4, smooth_factor=0.5, suffix=""):
        # this will retrieve the best trials, load the saved metrics and plot them in a single plot
        best_trials = self.oracle.get_best_trials(num_models)
        num_trials = len(best_trials)

        rows = (num_trials - 1) // wrap_columns + 1
        cols = min(num_trials, wrap_columns)

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5))
        # fig.subplots_adjust(hspace=0.4)
        # fig.tight_layout()

        for i, trial in enumerate(best_trials):
            trial_id = trial.trial_id
            trial_dir = self.get_trial_dir(trial_id)
            history_file = os.path.join(trial_dir, "history.json")

            with open(history_file, "r") as f:
                trial_history = json.load(f)

            ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
            for metric_name, metric_values in trial_history.items():
                ax.plot(metric_values, label=f"Trial {trial_id}: {metric_name}")

            ax.set_title(f"Trial {trial_id}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend()

        plt.savefig(os.path.join(self.oracle._project_dir, f"best_trials_metrics_{suffix}.png"))
        plt.close()
