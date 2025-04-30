#!/usr/bin/env python
import wandb
import logging
from datetime import timedelta
from tqdm import tqdm
import argparse
import re
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_timedelta(seconds):
    """Formats seconds into a human-readable string (days, hours, minutes, seconds)."""
    if seconds is None:
        return "N/A"
    delta = timedelta(seconds=int(seconds))
    return str(delta)

def get_wandb_runtime_summary(entity: str, project_filter_regex: Optional[str] = None):
    """
    Fetches runtimes from all projects for a given W&B entity,
    optionally filtering projects by regex.

    Args:
        entity: The W&B entity (team name) provided via command-line.
        project_filter_regex: Optional regex string to filter project names.

    Returns:
        A tuple containing:
        - total_runtime_seconds: Total runtime across all projects in seconds.
        - project_runtimes: A dictionary mapping project names to their total runtime in seconds.
    """
    api = wandb.Api()
    total_runtime_seconds = 0
    project_runtimes = {}

    try:
        logging.info(f"Fetching projects for entity: {entity}...")
        projects = api.projects(entity=entity)
        project_names = [project.name for project in projects]
        logging.info(f"Found {len(project_names)} projects.")

        # Filter projects if regex is provided
        if project_filter_regex:
            try:
                regex = re.compile(project_filter_regex)
                filtered_project_names = [name for name in project_names if regex.search(name)]
                logging.info(f"Filtering projects with regex: '{project_filter_regex}'. {len(filtered_project_names)} projects match.")
                project_names = filtered_project_names
            except re.error as e:
                logging.error(f"Invalid regex '{project_filter_regex}': {e}. Skipping filtering.")

    except Exception as e:
        logging.error(f"Failed to fetch projects for entity '{entity}': {e}")
        return 0, {}

    for project_name in tqdm(project_names, desc="Processing projects"):
        project_runtime = 0
        try:
            runs = api.runs(f"{entity}/{project_name}")
            logging.info(f"Processing project '{project_name}' with {len(runs)} runs...")
            for run in tqdm(runs, desc=f"Runs in {project_name}", leave=False):
                # _runtime is the total execution time in seconds stored in the summary
                runtime = run.summary.get("_runtime")
                if runtime is not None:
                    project_runtime += runtime
            project_runtimes[project_name] = project_runtime
            total_runtime_seconds += project_runtime
            logging.info(f"Project '{project_name}' total runtime: {format_timedelta(project_runtime)}")
        except Exception as e:
            logging.warning(f"Could not process project '{project_name}': {e}. Skipping.")

    return total_runtime_seconds, project_runtimes

if __name__ == "__main__":
    logging.info("Starting W&B runtime computation...")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Compute total GPU time for W&B projects.")
    parser.add_argument("-e", "--entity", type=str, required=True,
                        help="W&B entity (team name).")
    parser.add_argument("-f", "--filter", type=str, default=None,
                        help="Regex pattern to filter project names.")
    args = parser.parse_args()
    # ------------------------

    total_seconds, project_times = get_wandb_runtime_summary(args.entity, args.filter)

    logging.info("\n--- Runtime Summary ---")
    if not project_times:
        logging.warning("No projects or runs found, or failed to fetch data.")
    else:
        # Sort projects by name for consistent output
        sorted_projects = sorted(project_times.items())

        print("Per-Project GPU Time:")
        for project, seconds in sorted_projects:
            print(f"  - {project}: {format_timedelta(seconds)}")

        print(f"Total GPU Time Across All Projects: {format_timedelta(total_seconds)}")

    logging.info("Script finished.") 