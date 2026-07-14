"""
Manager agent: routes a task description to the right specialist agent
(CountGD for counting, StarDist for segmentation) and returns its result.

Neither agentic_countgd.py nor agentic_stardist.py exposes a plain run(image)
function -- each needs its own client/model (a Gradio Client for CountGD, a
loaded StarDist2D model for StarDist) plus, for CountGD, an Anthropic client
to turn the task description into a count target. ManagerAgent builds those
lazily on first use and calls each agent's existing run_countgd/run_stardist
function directly, so agentic_countgd.py and agentic_stardist.py stay
untouched.

Usage:
    python manager_agent.py --image cells.png --task "count the individual cells"
    python manager_agent.py --image tissue.png --task "segment the nuclei"
"""
import argparse

import anthropic
from gradio_client import Client
from stardist.models import StarDist2D

from agentic_countgd import COUNTGD_SPACE, interpret_prompt, run_countgd
from agentic_stardist import PRETRAINED_MODEL, load_image, run_stardist


class ManagerAgent:
    """Lazily builds the Claude/CountGD/StarDist clients it needs and reuses them across calls."""

    def __init__(self):
        self._claude = None
        self._countgd_client = None
        self._stardist_model = None

    @property
    def claude(self):
        if self._claude is None:
            self._claude = anthropic.Anthropic()
        return self._claude

    @property
    def countgd_client(self):
        if self._countgd_client is None:
            self._countgd_client = Client(COUNTGD_SPACE)
        return self._countgd_client

    @property
    def stardist_model(self):
        if self._stardist_model is None:
            self._stardist_model = StarDist2D.from_pretrained(PRETRAINED_MODEL)
        return self._stardist_model

    def select_agent(self, task_description: str) -> str:
        task = task_description.lower()
        if "segment" in task:
            return "stardist"
        if "count" in task:
            return "countgd"
        raise ValueError(
            f"Could not determine which agent to use for task: {task_description!r} "
            "(expected 'count' or 'segment' in the task description)"
        )

    def run(self, task_description: str, image_path: str) -> dict:
        agent = self.select_agent(task_description)

        if agent == "countgd":
            count_target = interpret_prompt(self.claude, task_description, image_path)
            annotated_image, predicted_count = run_countgd(self.countgd_client, image_path, count_target)
            return {
                "agent": "countgd",
                "count_target": count_target,
                "count": predicted_count,
                "annotated_image": annotated_image,
            }

        image = load_image(image_path)
        labels, details = run_stardist(self.stardist_model, image)
        return {
            "agent": "stardist",
            "num_nuclei": int(labels.max()),
            "labels": labels,
            "details": details,
        }


def main():
    parser = argparse.ArgumentParser(description="Route a task to CountGD or StarDist and run it")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--task", required=True, help="Task description, e.g. 'count the cells' or 'segment the nuclei'")
    args = parser.parse_args()

    manager = ManagerAgent()
    result = manager.run(args.task, args.image)

    print(f"[manager] routed to: {result['agent']}")
    if result["agent"] == "countgd":
        print(f"Count target: {result['count_target']!r}")
        print(f"Predicted count: {result['count']}")
    else:
        print(f"Detected nuclei: {result['num_nuclei']}")


if __name__ == "__main__":
    main()
