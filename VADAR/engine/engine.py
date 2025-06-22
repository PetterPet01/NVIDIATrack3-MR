import os
import re
import signal
import sys
import csv
import pandas
from typing import Literal

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from tqdm import tqdm
import json
import linecache
import runpy
import traceback
from engine.oracle import CLEVROracle
from engine.engine_utils import (
    Generator,
    get_methods_from_json,
    TimeoutException,
    timeout_handler,
)
from engine.predefined_modules import ModulesList


class Engine:
    def __init__(
        self,
        api_json=None,
        api_key_path="./api.key",
        oracle=CLEVROracle,
        results_folder_path="",
        models_path="",
        dataset="clevr",
    ):
        self.api_json = api_json
        self.oracle = oracle()
        self.results_folder_path = results_folder_path
        print("Initializing modules")
        self.modules_list = ModulesList(models_path=models_path, dataset=dataset, api_key_path=api_key_path)
        if api_json:
            self.api_methods, self.namespace = get_methods_from_json(self.api_json)
        else:
            self.api_methods = []
            self.namespace = {}

        self.namespace.update(self.modules_list.module_executes)
        self.trace_file_path = ""
        self.program_executable_path = ""
        self.result_file = ""
        self.execution_json = []
        self.namespace_line = sys.maxsize
        self.api_key_path = api_key_path
        self.output_csv_path = os.path.join(results_folder_path, "outputs.csv")

    def execute_programs(
        self, programs, questions, images_folder_path, oracle=False, scenes_json_path=""
    ):
        scenes_data = None
        if oracle:
            if os.path.exists(scenes_json_path):
                with open(scenes_json_path, "r") as file:
                    scenes_data = json.load(file)
                    scenes_data = scenes_data["scenes"]
            else:
                raise ValueError("Oracle requires scene data")

        folder_name = "program_execution"
        results_folder_path = os.path.join(
            self.results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path)

        for question, program_data in tqdm(
            zip(questions, programs), total=len(questions)
        ):
            question_results_path = os.path.join(
                results_folder_path,
                f"image_{question['image_index']}_question_{question['question_index']}/",
            )
            os.makedirs(question_results_path)
            exec_env_path = os.path.join(question_results_path, "exec_env/")

            
            os.makedirs(exec_env_path)

            self.trace_file_path = os.path.join(exec_env_path, "trace.html")
            with open(self.trace_file_path, "w+") as f:
                f.write(f"<h1>Question: {question['question']}</h1>")
            self.program_executable_path = os.path.join(
                exec_env_path, "executable_program.py"
            )
            self.result_file = os.path.join(exec_env_path, "result.json")

            image = Image.open(
                os.path.join(images_folder_path, question["image_filename"])
            )
            if scenes_data:
                scene_json = [
                    d
                    for d in scenes_data
                    if d.get("image_index") == program_data["image_index"]
                ]
                scene_json = scene_json[0]
                self.execution_json.append(
                    self.run_program(
                        program_data,
                        image,
                        question,
                        "oracle_execution",
                        scene_json,
                        error_count=0,
                    )
                )
            else:
                self.execution_json.append(
                    self.run_program(
                        program_data, image, question, "execution", error_count=0
                    )
                )

        execution_json_path = os.path.join(results_folder_path, "execution.json")

        with open(execution_json_path, "w+") as file:
            json.dump(self.execution_json, file)

        if oracle:
            self.save_evaluation_accuracy(
                self.execution_json, results_folder_path, "oracle_execution"
            )
        else:
            self.save_evaluation_accuracy(
                self.execution_json, results_folder_path, "execution"
            )

    def retry_executions_with_oracle(
        execution_folder_path, questions, images_folder_path, scenes_json_path
    ):
        engine = Engine(None, results_folder_path=execution_folder_path)

        for question in tqdm(questions):
            question_results_path = os.path.join(
                execution_folder_path,
                f"image_{question['image_index']}_question_{question['question_index']}/",
            )
            exec_env_path = os.path.join(question_results_path, "exec_env/")

            engine.trace_file_path = os.path.join(exec_env_path, "trace.html")
            engine.program_executable_path = os.path.join(
                exec_env_path, "executable_program.py"
            )
            engine.result_file = os.path.join(exec_env_path, "result.json")

            with open(engine.program_executable_path, "r") as file:
                program_data = file.read()
            program_data = program_data.replace(
                "/data/damiano/clevr/results1/", "/data/rohun/"
            )
            with open(engine.program_executable_path, "w") as file:
                file.write(program_data)

            # get image and scene data
            image = Image.open(
                os.path.join(images_folder_path, question["image_filename"])
            )
            if os.path.exists(scenes_json_path):
                with open(scenes_json_path, "r") as file:
                    scenes_data = json.load(file)
                    scenes_data = scenes_data["scenes"]
            else:
                raise ValueError("Oracle requires scene data")

            scene_json = [
                d
                for d in scenes_data
                if d.get("image_index") == question["image_index"]
            ]
            scene_json = scene_json[0]

            # execute file
            engine.modules_list.set_trace_path(engine.trace_file_path)
            execution_data = {}

            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            engine.namespace.update(image=image)
            engine.modules_list.set_oracle(engine.oracle, image, scene_json)
            error = engine._execute_file()

            try:
                with open(engine.result_file, "r") as f:
                    result_namespace = json.load(f)
            except Exception as e:
                result_namespace = {"final_result": f"Error: {error}"}

            execution_data["oracle_execution"] = {}
            execution_data["oracle_execution"]["question"] = question
            execution_data["oracle_execution"]["result_namespace"] = result_namespace
            if "final_result" in result_namespace:
                execution_data["oracle_execution"]["answer"] = result_namespace[
                    "final_result"
                ]
            else:
                execution_data["oracle_execution"]["answer"] = ""
            engine.modules_list.clear_oracle()
            engine.execution_json.append(execution_data)

        execution_json_path = os.path.join(
            execution_folder_path, "execution_oracle.json"
        )

        with open(execution_json_path, "w+") as file:
            json.dump(engine.execution_json, file)

        engine.save_evaluation_accuracy(
            engine.execution_json, execution_folder_path, "oracle_execution"
        )

    def run_program(
        self,
        program_data,
        image,
        question,
        execution_type=Literal["execution", "oracle_execution"],
        scene_json=None,
        error_count=0,
    ):
        program = program_data["program"]

        self.modules_list.set_trace_path(self.trace_file_path)
        execution_data = {}

        image = image.convert("RGB")
        self.namespace.update(image=image)

        try:
            if isinstance(program, list):
                program = program[0]
        except Exception as e:
            if error_count < 5:
                print("No program found")
                corrected_program_data = self.correct_program_error(
                    program_data, Exception("No program found"), question
                )
                return self.run_program(
                    corrected_program_data,
                    image,
                    question,
                    execution_type,
                    scene_json,
                    error_count + 1,
                )
            else:
                program = ""
        self._add_program_to_file(program)

        if scene_json:
            self.modules_list.set_oracle(self.oracle, image, scene_json)

        error = self._execute_file()
        if error and error_count < 5:
            corrected_program_data = self.correct_program_error(
                program_data, error, question
            )
        
            if os.path.exists(self.trace_file_path):
                os.remove(self.trace_file_path)
            
            return self.run_program(
                corrected_program_data,
                image,
                question,
                execution_type,
                scene_json,
                error_count + 1,
            )

        try:
            with open(self.result_file, "r") as f:
                result_namespace = json.load(f)
        except Exception as e:
            result_namespace = {"final_result": f"Error: {error}"}

        execution_data[execution_type] = {}
        execution_data[execution_type]["question"] = question
        execution_data[execution_type]["program"] = program
        execution_data[execution_type]["result_namespace"] = result_namespace
        if "final_result" in result_namespace:
            final_result = result_namespace["final_result"]
            if isinstance(final_result, bool):
                execution_data[execution_type]["answer"] = (
                    "yes" if final_result else "no"
                )
            elif isinstance(final_result, str):
                execution_data[execution_type]["answer"] = final_result.lower()
            else:
                execution_data[execution_type]["answer"] = final_result
        else:
            execution_data[execution_type]["answer"] = ""

        if scene_json:
            self.modules_list.clear_oracle()
        return execution_data

    def write_csv(self, filename, entries):
        fields = [
            "question",
            "image_index",
            "answer_type",
            "ground_truth",
            "prediction",
        ]
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def correct_program_error(self, program_data, error, question):
        messages = program_data["messages"]
        messages.append(
            {
                "role": "user",
                "content": f"\n There was an error in running the code: {error}. Try again and include the program between <program></program>",
            }
        )
        generator = InternVLGenerator(
            model_path=program_data["model_name"], precision="bf16", use_8bit=False, multi_gpu=False
        )
        output, messages = generator.generate(None, messages)
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        program = re.findall(r"<program>(.*?)</program>", output, re.DOTALL)
        program_data = {
            "image_index": question["image_index"],
            "question_index": question["question_index"],
            "program": program,
            "prompt": program_data["prompt"],
            "output": output,
            "messages": messages,
            "model_name": generator.model_name,
        }
        return program_data

    def write_summarized_results(self, csv_path, results_pth):
        mra_thresholds = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        correct_at_threshold = {key: 0 for key in mra_thresholds}
        yn_correct = 0
        yn_n = 0
        num_ct_n = 0
        num_ct_correct = 0
        multi_correct = 0
        multi_n = 0
        num_other_n = 0

        reader = csv.reader(open(csv_path), delimiter=",")
        headers = next(reader, None)

        for row in reader:
            ans_type = row[-3]
            gt = row[-2]
            pred = row[-1]

            # Numeric (count)
            if ans_type == "int":
                num_ct_n += 1
                try:
                    pred = int(pred)
                except:
                    continue
                gt = int(gt)
                if gt == pred:
                    num_ct_correct += 1
            elif ans_type == "str":
                # Yes/No
                if gt in ["yes", "no"]:
                    yn_n += 1
                    try:
                        if gt == pred.lower():
                            yn_correct += 1
                    except:
                        continue
                # multi
                else:
                    multi_n += 1
                    try:
                        if gt == pred.lower():
                            multi_correct += 1
                    except:
                        continue
            elif ans_type == "float":
                # Numeric (other)
                num_other_n += 1
                for threshold in mra_thresholds:
                    try:
                        pred = float(pred)
                    except:
                        continue
                    gt = float(gt)
                    if abs(gt - pred) / gt < threshold:
                        correct_at_threshold[threshold] += 1

        # Compute AVG Accuracies
        yn_acc = yn_correct / yn_n if yn_n != 0 else None
        multi_acc = multi_correct / multi_n if multi_n != 0 else None
        num_ct_acc = num_ct_correct / num_ct_n if num_ct_n != 0 else None
        num_other_mra = 0

        if num_other_n != 0:
            for threshold in mra_thresholds:
                correct_at_threshold[threshold] /= num_other_n
                num_other_mra += correct_at_threshold[threshold]

            num_other_mra = num_other_mra / len(mra_thresholds)
        else:
            num_other_mra = None

        with open(results_pth, "w") as f:
            f.write("-------- Summary Results --------\n")
            f.write(f"Yes/No Accuracy: {yn_acc}\n")
            f.write(f"Multiple Choice Accuracy: {multi_acc}\n")
            f.write(f"Numeric (count) Accuracy: {num_ct_acc}\n")
            f.write(f"Numeric (other) MRA: {num_other_mra}")

    def save_evaluation_accuracy(
        self,
        execution_data: list,
        results_path: str,
        execution_type: Literal["execution", "oracle_execution"],
    ):
        execution_results_path = os.path.join(results_path, execution_type + ".txt")
        execution_sheet_path = os.path.join(results_path, execution_type + ".csv")
        execution_results = ""
        execution_results += "\nResults:\n"
        num_correct = 0
        num_correct_oracle = 0
        only_oracle_correct = 0
        only_execution_correct = 0
        both_correct = 0

        json_results = []

        for execution in execution_data:
            json_results.append(
                {
                    "question": str(execution[execution_type]["question"]["question"]),
                    "image_index": execution[execution_type]["question"]["image_index"],
                    "answer_type": (
                        str(execution[execution_type]["question"]["answer_type"])
                        if "answer_type" in execution[execution_type]["question"]
                        else ""
                    ),
                    "ground_truth": (
                        str(execution[execution_type]["question"]["answer"])
                        if "answer" in execution[execution_type]["question"]
                        else ""
                    ),
                    "prediction": str(execution[execution_type]["answer"]),
                }
            )

            execution_results += (
                f"Image: {execution[execution_type]['question']['image_index']}\n"
            )
            execution_results += f"Question {execution[execution_type]['question']['question_index']}: {execution[execution_type]['question']['question']}\n"
            if "program" in execution[execution_type]:
                execution_results += (
                    f"Program: {execution[execution_type]['program']}\n"
                )
            execution_results += (
                f"Predicted Answer: {execution[execution_type]['answer']}\n"
            )
            # execution_results += (
            #     f"Predicted Oracle Answer: {execution['oracle_execution']['answer']}\n"
            # )
            execution_results += (
                f"Correct Answer: {execution[execution_type]['question']['answer']}\n"
            )

            if isinstance(execution[execution_type]["answer"], bool):
                answer = "yes" if execution[execution_type]["answer"] else "no"
                if answer == execution[execution_type]["question"]["answer"]:
                    num_correct += 1
            elif (
                str(execution[execution_type]["answer"])
                == execution[execution_type]["question"]["answer"]
            ):
                num_correct += 1
            #     if (
            #         str(execution["oracle_execution"]["answer"])
            #         == execution["execution"]["question"]["answer"]
            #     ):
            #         num_correct_oracle += 1
            #         both_correct += 1
            #     else:
            #         only_execution_correct += 1
            # elif (
            #     str(execution["oracle_execution"]["answer"])
            #     == execution["execution"]["question"]["answer"]
            # ):
            #     num_correct_oracle += 1
            #     only_oracle_correct += 1

        execution_results += f"\nOnly Execution Correct: {float(only_execution_correct)/len(execution_data)}\n"
        # execution_results += (
        #     # f"Only Oracle Correct: {float(only_oracle_correct)/len(execution_data)}\n"
        # )
        execution_results += (
            f"Both Correct: {float(both_correct)/len(execution_data)}\n"
        )
        # execution_results += f"Both Incorrect: {1 - (float(both_correct + only_oracle_correct + only_execution_correct)/len(execution_data))}\n"
        execution_results += f"Accuracy: {float(num_correct)/len(execution_data)}\n"
        # execution_results += (
        #     f"Oracle Accuracy: {float(num_correct_oracle)/len(execution_data)}\n\n"
        # )

        with open(execution_results_path, "w+") as file:
            file.write(execution_results)
            file.close()

        csv_path = os.path.join(results_path, f"{execution_type}.csv")
        self.write_csv(csv_path, json_results)

        # Write summarized results
        self.write_summarized_results(
            csv_path, os.path.join(results_path, "results.txt")
        )

    def _add_program_to_file(self, program):
        with open(self.program_executable_path, "w") as file:
            file.write("import math\n")
            file.writelines(f"{method}\n" for method in self.api_methods)
            file.write("\n# PROGRAM STARTS HERE\n")

        new_program_content = [f"{line}\n" for line in program.split("\n")]

        write_namespace_code = f"""
# WRITE NAMESPACE
import json

def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {{k: v for k, v in globals().items() if is_serializable(v)}}

with open("{self.result_file}", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        """

        with open(self.program_executable_path, "a") as file:
            file.writelines(new_program_content)
            file.write(write_namespace_code)

    def _trace_execution(self, frame, event, arg):
        if event == "line":
            filename = frame.f_globals.get("__file__", None)
            if filename and os.path.basename(filename) == os.path.basename(self.program_executable_path):
                lineno = frame.f_lineno
                line = linecache.getline(filename, lineno).strip()
                if lineno > self.namespace_line:
                    return self._trace_execution
                if "import math" in line:
                    return self._trace_execution
                if "import" in line:
                    self.namespace_line = lineno
                    return self._trace_execution
                # Get function name if we're inside one
                function_name = frame.f_code.co_name
                trace_line = f"<p>{lineno}: "
                if function_name and function_name != '<module>':
                    trace_line += f"[In method {function_name}] "
                trace_line += f"<code>{line}</code></p>\n"
                with open(self.trace_file_path, "a+") as f:
                    f.write(trace_line)
        return self._trace_execution

    def _execute_file(self):
        sys.settrace(self._trace_execution)
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(200)
            runpy.run_path(self.program_executable_path, init_globals=self.namespace)
            signal.alarm(0)
        except TimeoutException as e:
            return e
        except Exception as e:
            return e
        finally:
            sys.settrace(None)
        return

