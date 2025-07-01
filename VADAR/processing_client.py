# distributed_vadar/processing_client.py - FIXED VERSION
import os
import time
import requests
import json
import traceback
import argparse
from typing import List, Dict, Any, Optional
import concurrent.futures
from concurrent.futures import TimeoutError

# Import the refactored library and necessary components
from full_pipeline_refactored import (
    VADARContext,
    initialize_modules,
    initialize_and_get_generator,
    process_annotation_by_index,
    process_query_stage2,
    GeneralizedSceneGraphGenerator,
)
from rich.console import Console

# --- Configuration (Unaltered, as requested) ---
SERVER_URL = "http://159.223.81.229:8000"
MISTRAL_URL = "http://159.223.81.229:8001/v1"
BATCH_SIZE = 16
MAX_WORKERS = 8
# The base directory where the 'images' folder is located on the client machine.
IMAGE_BASE_DIR = "/root/Projects/NVIDIAFinalRun/PhysicalAI_Dataset/val"
# FIX: Define a timeout for individual task processing to prevent hanging
TASK_PROCESSING_TIMEOUT = 180  # 3 minutes, slightly more than server timeout

console = Console()

answer_types_json = '/content/NVIDIATrack3-MR/VADAR/demo-notebook/answer_types.json'


def expand_query(json_path, query_id):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for item in data:
        if item['id'] == query_id:
            print('EXPANDING QUERY IN PROCESSING CLIENT.')
            if item['type'] == 'string':
                return "This IS a left_right question. Answer ONLY with 'left' or 'right'"
            elif item['type'] == 'number':
                return "This IS either a <regionX>, COUNT, or DISTANCE question. Answer ONLY in **number** (integer or float)"
            else:
                return f"Unknown type: {item['type']}"

    return ""


# <<< UNCHANGED process_single_task FUNCTION >>>
def process_single_task(
        task_data: Dict[str, Any],
        context: VADARContext,
        sgg_generator: GeneralizedSceneGraphGenerator,
        image_base_dir: str
) -> Optional[Dict[str, Any]]:
    task_id = task_data.get('id', task_data.get('image', 'UnknownID'))
    console.print(f"[Task {task_id}] Starting processing...")

    try:
        task_data_for_pipeline = task_data.copy()
        question = "N/A"
        if "conversations" in task_data_for_pipeline and task_data_for_pipeline["conversations"]:
            for convo in task_data_for_pipeline["conversations"]:
                if convo.get("from") == "human":
                    question = convo.get("value", "N/A")
                    break
        task_data_for_pipeline['question'] = question + expand_query(answer_types_json, task_id)
        images_directory = os.path.join(image_base_dir, "images")

        processed_data_stage1 = process_annotation_by_index(
            context,
            sgg_generator,
            [task_data_for_pipeline],
            images_directory,
            index=0
        )
        if not processed_data_stage1:
            console.print(f"[Task {task_id}] [bold red]Error:[/bold red] Stage 1 (SGG) failed.")
            return None

        result = process_query_stage2(context, processed_data_stage1)
        if not result:
            console.print(f"[Task {task_id}] [bold red]Error:[/bold red] Stage 2 (Agentic) failed.")
            return None

        console.print(
            f"[Task {task_id}] [bold green]Successfully processed.[/bold green] Result: {result.get('normalized_answer', 'N/A')}")
        return result

    except Exception as e:
        console.print(f"[Task {task_id}] [bold red]FATAL ERROR during processing:[/bold red]")
        console.print(traceback.format_exc())
        return None


# <<< MODIFIED submit_batch_results FUNCTION >>>
def submit_batch_results(server_url: str, results: List[Dict[str, Any]]):
    """
    Submits a list of results (both success and failure) to the server.
    """
    if not results:
        console.print("[yellow]No results (success or failure) to submit for this batch. This is unusual.[/yellow]")
        return

    console.print(f"\n[Phase 3] Submitting {len(results)} results for the completed batch...")
    try:
        # The server expects a list of objects with 'id' and 'normalized_answer'
        # Our success/failure objects already match this TaskResult pydantic model
        submit_response = requests.post(f"{server_url}/submit_results", json=results, timeout=30)
        submit_response.raise_for_status()
        console.print("[bold green]Batch results submitted successfully![/bold green]")
        # You can optionally print the server's response for more detail
        # console.print(f"Server response: {submit_response.json()}")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error submitting results for this batch: {e}[/bold red]")
        console.print("[yellow]These tasks will remain IN_PROGRESS on the server until they time out.[/yellow]")


# <<< HEAVILY MODIFIED run_client FUNCTION >>>
def run_client(server_url: str, mistral_url: str, image_dir: str):
    """The main loop for the worker client."""
    console.print("=" * 50, style="bold cyan")
    console.print(" VADAR Processing Client Initializing... ", style="bold cyan")
    console.print("=" * 50, style="bold cyan")
    console.print(f"Server URL: [yellow]{server_url}[/yellow]")
    console.print(f"Image Base Directory: [yellow]{image_dir}[/yellow]")

    console.print("\n[Phase 1] Loading all models. This may take a moment...")
    try:
        context = initialize_modules(qwen_api_base_url=mistral_url)
        sgg_generator = initialize_and_get_generator(context)
        if not context or not sgg_generator:
            raise RuntimeError("Failed to initialize models.")
        console.print("[bold green]Models loaded successfully.[/bold green]")
    except Exception as e:
        console.print("[bold red]FATAL: Could not initialize models. Exiting.[/bold red]")
        console.print(traceback.format_exc())
        return

    while True:
        try:
            console.print(f"\n[Phase 2] Requesting a batch of {BATCH_SIZE} tasks...")
            response = requests.get(f"{server_url}/get_batch", params={"size": BATCH_SIZE}, timeout=20)
            response.raise_for_status()
            tasks_to_process = response.json()

            if not tasks_to_process:
                console.print("[yellow]No work available. Waiting for 30 seconds...[/yellow]")
                time.sleep(30)
                continue

            console.print(f"--> Received {len(tasks_to_process)} tasks. Starting batch processing...")

            # FIX 1: Keep track of ALL task IDs checked out in this batch.
            original_task_ids = {task['id'] for task in tasks_to_process}
            all_results_for_submission = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task_id = {
                    executor.submit(process_single_task, task, context, sgg_generator, image_dir): task['id']
                    for task in tasks_to_process
                }

                for future in concurrent.futures.as_completed(future_to_task_id):
                    task_id = future_to_task_id[future]
                    try:
                        # FIX 2: Add a timeout to getting the result to prevent hangs.
                        result = future.result(timeout=TASK_PROCESSING_TIMEOUT)
                        if result:
                            # This is a successful result
                            all_results_for_submission.append(result)
                        else:
                            # The function returned None, indicating a graceful failure.
                            console.print(
                                f"[Task {task_id}] [yellow]Processing failed gracefully. Reporting failure.[/yellow]")
                            all_results_for_submission.append(
                                {"id": task_id, "normalized_answer": "CLIENT_PROCESSING_FAILED"})

                    except TimeoutError:
                        console.print(f"[Task {task_id}] [bold red]Processing TIMED OUT. Reporting failure.[/bold red]")
                        all_results_for_submission.append(
                            {"id": task_id, "normalized_answer": "CLIENT_PROCESSING_TIMEOUT"})

                    except Exception as exc:
                        # This catches exceptions that happened inside the future.
                        console.print(
                            f"[Task {task_id}] [bold red]Generated an exception: {exc}. Reporting failure.[/bold red]")
                        all_results_for_submission.append({"id": task_id, "normalized_answer": "CLIENT_EXCEPTION"})

            # FIX 3: Reconcile the original batch with what was processed to catch any gaps.
            # This is a safety net in case the loop logic has an issue.
            processed_ids = {res['id'] for res in all_results_for_submission}
            unaccounted_ids = original_task_ids - processed_ids
            if unaccounted_ids:
                console.print(
                    f"[bold yellow]WARNING: Found {len(unaccounted_ids)} unaccounted-for tasks. Reporting as failures.[/bold yellow]")
                for task_id in unaccounted_ids:
                    all_results_for_submission.append({"id": task_id, "normalized_answer": "CLIENT_LOGIC_ERROR"})

            console.print(
                f"--> Batch processing complete. Reporting {len(all_results_for_submission)} results for {len(original_task_ids)} tasks.")

            # Submit ALL results, both success and failure, to clear the queue.
            submit_batch_results(server_url, all_results_for_submission)

        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Cannot connect to server. Retrying in 60s... Error: {e}[/bold red]")
            time.sleep(60)
        except Exception as e:
            console.print(f"[bold red]Unexpected client error:[/bold red]")
            traceback.print_exc()
            console.print("[yellow]Restarting loop in 60s...[/yellow]")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the VADAR Processing Client.")
    parser.add_argument("--server_url", type=str, default=SERVER_URL, help="URL of the task server.")
    parser.add_argument("--mistral_url", type=str, default=MISTRAL_URL,
                        help="URL for the Qwen/Mistral compatible API endpoint.")
    parser.add_argument("--image_dir", type=str, default=IMAGE_BASE_DIR,
                        help="Base directory of the image dataset on this client machine.")
    args = parser.parse_args()

    run_client(server_url=args.server_url, mistral_url=args.mistral_url, image_dir=args.image_dir)