import os
import sys
import json
import base64
import io
import re
from tqdm import tqdm
from PIL import Image
import signal
import linecache
import runpy
import shutil
import traceback

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from engine.engine_utils import (
    Generator,
    correct_indentation,
    replace_tabs_with_spaces,
    untab,
    TimeoutException,
    timeout_handler,
)
from prompts.signature_prompt import SIGNATURE_PROMPT, SIGNATURE_PROMPT_CLEVR
from prompts.program_prompt import (
    PROGRAM_PROMPT,
    PROGRAM_PROMPT_GQA,
    PROGRAM_PROMPT_CLEVR,
)
from prompts.api_prompt import API_PROMPT, API_PROMPT_GQA, API_PROMPT_CLEVR


class Agent:
    def __init__(
        self,
        model_name="gpt-4o",
        write_results=True,
        api_key_path="./api.key",
        dataset="clevr",
    ):
        self.generator = InternVLGenerator(model_path=model_name, precision="bf16", use_8bit=False, multi_gpu=False)
        self.write_results = write_results
        self.dataset = dataset


class SignatureAgent(Agent):

    def __init__(
        self, predef_signatures, model_name="gpt-4o", write_results=True, headers=[]
    ):
        super().__init__(model_name, write_results)
        self.signatures = predef_signatures
        self.predef_signatures = predef_signatures
        self.headers = headers
        self.generated_docstrings = []
        self.generated_signatures = []
        self.generated_headers = []
        self.method_names = []

        for header in headers:
            self.generated_docstrings.append(header["docstring"])
            self.generated_signatures.append(header["signature"])
            self.generated_headers.append(header["docstring"] + header["signature"])

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def __call__(self, questions, prompt):
        output, _ = self.generator.generate(
            prompt.format(signatures=self.signatures, question="\n\n".join(questions))
        )
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")

        docstrings = re.findall(r"<docstring>(.*?)</docstring>", output, re.DOTALL)
        signatures = re.findall(r"<signature>(.*?)</signature>", output, re.DOTALL)

        self.generated_docstrings += docstrings
        self.generated_signatures += signatures
        headers = [doc + sig for doc, sig in zip(docstrings, signatures)]
        method_names = [
            re.compile(r"def (\w+)\s*\(.*\):").search(sig).group(1)
            for sig in signatures
        ]
        self.method_names += method_names
        self.generated_headers += headers
        self.signatures += "\n\n".join(headers)
        self.headers += [
            {"method_name": method_name, "docstring": doc, "signature": sig}
            for doc, sig, method_name in zip(docstrings, signatures, method_names)
        ]

        return headers, output

    def signatures_info(self):
        return [
            {"docstring": doc, "signature": sig}
            for doc, sig in zip(self.generated_docstrings, self.generated_signatures)
        ]

    def get_signatures(
        self,
        questions_data,
        images_folder_path,
        results_folder_path,
        prompt=None,
        question_batch_size=10,
    ):
        # Use dataset-specific prompt if none provided
        if self.dataset in ["clevr", "gqa"]:
            prompt = SIGNATURE_PROMPT_CLEVR
        else:
            prompt = SIGNATURE_PROMPT

        folder_name = "signature_generator"
        results_folder_path = os.path.join(
            results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path)

        question_batches = []
        for i in range(0, len(questions_data), question_batch_size):
            question_batches.append(questions_data[i : i + question_batch_size])

        for question_batch in tqdm(question_batches):

            questions = [question_data["question"] for question_data in question_batch]
            prompt_text = prompt.format(
                signatures=self.signatures, question="\n\n".join(questions)
            )
            headers, output = self(questions, prompt)

            for question_data in question_batch:

                html_path = os.path.join(
                    results_folder_path,
                    f"image_{question_data['image_index']}_question_{question_data['question_index']}.html",
                )

                if self.write_results:
                    with open(html_path, "wb+") as file:

                        # get image
                        image = Image.open(
                            os.path.join(
                                images_folder_path, question_data["image_filename"]
                            )
                        )
                        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                        rgb_image = image.convert("RGB")
                        image_io = io.BytesIO()
                        rgb_image.save(image_io, format="PNG")
                        image_bytes = base64.b64encode(image_io.getvalue()).decode(
                            "ascii"
                        )

                        # Write question and image
                        file.write(
                            (f"<h1>{question_data['question']}</h1>\n").encode("utf-8")
                        )
                        file.write(
                            (
                                f"<img src='data:image/jpeg;base64,{image_bytes}'>\n"
                            ).encode("utf-8")
                        )

                        # prompt
                        file.write((f"<h1>Prompt</h1>\n").encode("utf-8"))
                        file.write(
                            (
                                f"<code>{prompt_text}</code>\n".replace("\n", "<br>")
                            ).encode("utf-8")
                        )

                        file.write((f"<h1>LLM Output</h1>\n").encode("utf-8"))
                        file.write(
                            (f"<code>{output}</code>\n".replace("\n", "<br>")).encode(
                                "utf-8"
                            )
                        )

                        new_signatures = "\n\n".join(headers)
                        file.write((f"<h1>New Signatures</h1>\n").encode("utf-8"))
                        file.write(
                            (
                                f"<code>{new_signatures}</code>\n".replace("\n", "<br>")
                            ).encode("utf-8")
                        )

                        file.close()

            signatures_path = os.path.join(results_folder_path, "signatures.json")

            signatures_info = self.signatures_info()

            with open(signatures_path, "w+") as file:
                json.dump(signatures_info, file)

        return signatures_path, signatures_info


class APIAgent(Agent):

    def __init__(
        self, signature_agent, dataset, model_name="gpt-4o", write_results=True, api=[]
    ):
        super().__init__(model_name, write_results)
        self.signature_agent = signature_agent
        self.dataset = dataset
        self.implementations = []
        self.api = api
        self.error_counts = [0 for _ in range(len(self.signature_agent.method_names))]
        self.namespace = {}
        self.namespace_line = sys.maxsize
        self.trace_file_path = ""
        self.implemented = [
            False for _ in range(len(self.signature_agent.method_names))
        ]
        self.method_stack = []
        self.max_num_errors = 5
        self.pbar = tqdm(total=len(self.signature_agent.method_names))

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def __call__(
        self, method_name, docstring, signature, results_folder_path, prompt=None, api_info=None
    ):
        # Use dataset-specific prompt if none provided
        # if prompt is None:
        #     prompt = API_PROMPT
        if prompt is None:
            if self.dataset == "clevr":
                prompt = API_PROMPT_CLEVR
            elif self.dataset == "gqa":
                prompt = API_PROMPT_GQA
            else:
                prompt = API_PROMPT

        """
        Generate an implementation for the given method name, docstring, and signature.
        If the method has already been implemented, return the existing implementation.
        """
        if self.implemented[self.signature_agent.method_names.index(method_name)]:
            return [
                api_info
                for api_info in self.api
                if api_info.get("method_name") == method_name
            ][0]["implementation"], ""

        
        self.pbar.set_description(
            f"Implementing {method_name} at error count {self.error_counts[self.signature_agent.method_names.index(method_name)]}"
        )
        # Get the generated signatures for all methods except the current one
        generated_signatures = [
            header["docstring"] + header["signature"]
            for header in self.signature_agent.headers
            if header["method_name"] != method_name and
            self.error_counts[self.signature_agent.method_names.index(header["method_name"])] < self.max_num_errors
        ]
        generated_signatures = "\n\n".join(generated_signatures)

        if api_info:
            messages = api_info["messages"] if api_info["messages"] is not None else None
            output, messages = self.generator.generate(
                "",
                messages=messages
            )
        else:
            messages = None
            output, messages = self.generator.generate(
                prompt.format(
                    predef_signatures=self.signature_agent.predef_signatures,
                    generated_signatures=generated_signatures,
                    docstring=docstring,
                    signature=signature,
                ),
                messages=messages
            )

        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")

        # Extract the implementation from the output
        implementation = re.findall(
            r"<implementation>(.*?)</implementation>", output, re.DOTALL
        )
        implementation = implementation[0]

        # Remove the function signature and unindent the code
        lines = implementation.split("\n")
        signature_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                signature_index = i
                break
        if signature_index is not None:
            implementation = "\n".join(lines[signature_index + 1 :])
        else:
            implementation = correct_indentation(implementation)
            

        api_info = {
            "docstring": docstring,
            "signature": signature,
            "implementation": replace_tabs_with_spaces(implementation),
            "method_name": method_name,
            "messages": messages,
            "error": None,
        }

        # Test the implementation
        api_info = self.test_implementation(api_info, results_folder_path, prompt=prompt)

        if api_info["error"]:
            # If the implementation fails, recurse on the method
            if (
                self.error_counts[self.signature_agent.method_names.index(method_name)]
                < self.max_num_errors
            ):
                api_info["messages"].append({"role": "user", "content": f"The implementation failed with error:\n{api_info['error']}\nPlease fix the implementation by rewriting the full method again in <implementation></implementation> tags."})
                self.method_stack.append(method_name)
                self(
                    method_name,
                    docstring,
                    signature,
                    results_folder_path,
                    prompt=prompt,
                    api_info=api_info
                )
        else:
            # If the implementation succeeds, add it to the API and return it
            self.api.append(api_info)
            self.implementations.append(implementation)
            return implementation, output
        return "", ""

    def test_implementation(self, api_info, results_folder_path, prompt=API_PROMPT):
        method_name = (
            re.compile(r"def (\w+)\s*\(.*\):").search(api_info["signature"]).group(1)
        )
        if self.error_counts[self.signature_agent.method_names.index(method_name)] >= 5:
            api_info["error"] = "Implementation failed"
            return api_info
        
        predef_api = []

        # Get the predefined API
        for signature_text in self.signature_agent.predef_signatures.split('\n\n"""'):
            signature_start = signature_text.find("def ")

            docstring = signature_text[:signature_start].strip()
            signature = signature_text[signature_start:].strip()

            _, returns = self._get_docstring_types(docstring)

            implementation = self._get_return_code(returns)

            predef_api.append(
                {
                    "docstring": docstring,
                    "signature": signature,
                    "implementation": implementation,
                }
            )

        implementation_results_path = os.path.join(
            results_folder_path, f"{method_name}"
        )
        if os.path.exists(implementation_results_path):
            shutil.rmtree(implementation_results_path)
        os.makedirs(implementation_results_path)
        exec_env_path = os.path.join(implementation_results_path, "exec_env/")
        os.makedirs(exec_env_path)

        self.trace_file_path = os.path.join(exec_env_path, "trace.html")
        program_executable_path = os.path.join(exec_env_path, "executable_program.py")
        result_file = os.path.join(exec_env_path, "result.json")

        # Write the predefined API to the executable program
        for method_info in predef_api:
            with open(program_executable_path, "a+") as f:
                f.write("import math\n")
                f.write(method_info["signature"] + "\n")
                f.write(method_info["implementation"] + "\n\n")

        # Get the arguments and return types of the method
        arg_types, returns = self._get_docstring_types(api_info["docstring"])

        # Create the namespace that will be used to execute the program
        self.namespace = {}

        # Initialize the arguments of the method
        if self.dataset in ["clevr", "gqa"]:
            for arg, type in arg_types:
                if type == "image":
                    self.namespace.update({arg: Image.new("RGB", (1000, 500), "white")})
                elif type == "int":
                    self.namespace.update({arg: 25})
                elif type == "string":
                    self.namespace.update({arg: ""})
                elif type == "float":
                    self.namespace.update({arg: 1.0})
                elif type == "list":
                    self.namespace.update({arg: [[25, 25]]})
                elif type == "tuple":
                    self.namespace.update({arg: (50, 50)})
                else:
                    self.namespace.update({arg: 1})
        else:
            for arg, type in arg_types:
                if type == "image":
                    self.namespace.update({arg: Image.new("RGB", (1000, 500), "white")})
                elif type == "int":
                    self.namespace.update({arg: 25})
                elif type == "string":
                    self.namespace.update({arg: ""})
                elif type == "float":
                    self.namespace.update({arg: 1.0})
                elif type == "list":
                    self.namespace.update({arg: [25, 25, 50, 50]})
                elif type == "tuple":
                    self.namespace.update({arg: (50, 50)})
                else:
                    self.namespace.update({arg: 1})

        # Write the method to the executable program
        with open(program_executable_path, "a+") as file:
            for method_info in self.api:
                file.write(method_info["signature"] + "\n")
                file.write(method_info["implementation"] + "\n\n")
            file.write("\n# PROGRAM STARTS HERE\n")
            lines = untab(api_info["implementation"]).split("\n")
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith("return "):
                    modified_line = line.replace("return", "final_result =", 1)
                    file.write(modified_line + "\n")
                else:
                    file.write(line + "\n")
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

with open("{result_file}", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        """
            file.write(write_namespace_code)

        # Execute the program and get the result
        result = self._execute_file(program_executable_path)
        if result:
            error, stacktrace = result
        else:
            error = None
            stacktrace = None

        if error:
            error = str(error)
            stacktrace = str(stacktrace)

            # Feed error back to the model and try again
            method_name = re.compile(r"def (\w+)\s*\(.*\):").search(api_info["signature"]).group(1)
            print(f"Error in executing {method_name}: {error}")

            # Check for undefined method error first
            undefined_method = re.search(r"name '(\w+)' is not defined", error)
            if undefined_method:
                undefined_method = undefined_method.group(1)
                try:
                    # Check for infinite recursion
                    if (
                        len(self.method_stack) > 4
                        and self.method_stack[-2] == undefined_method
                        and self.method_stack[-3] == method_name
                        and self.method_stack[-4] == undefined_method
                    ):
                        print("Infinite recursion detected")
                        # Mark both methods as failed
                        self.error_counts[
                            self.signature_agent.method_names.index(undefined_method)
                        ] = self.max_num_errors
                        self.error_counts[
                            self.signature_agent.method_names.index(method_name)
                        ] = self.max_num_errors
                        api_info["error"] = "Implementation failed"
                        return api_info
                    elif undefined_method == method_name:
                        # If the undefined method is the same as the current method, mark it as failed
                        self.error_counts[
                            self.signature_agent.method_names.index(method_name)
                        ] = self.max_num_errors
                        api_info["error"] = "Implementation failed"
                        return api_info

                    # Recursively call the generate method on the undefined method
                    method_name_index = self.signature_agent.method_names.index(
                        undefined_method
                    )
                    header = self.signature_agent.headers[method_name_index]
                    self.method_stack.append(undefined_method)
                    self(
                        undefined_method,
                        header["docstring"],
                        header["signature"],
                        results_folder_path,
                        prompt=prompt,
                    )
                    return self.test_implementation(
                        api_info, results_folder_path, prompt=prompt
                    )
                except ValueError:
                    # If the undefined method is not found in the signature agent, increment the error count
                    self.error_counts[
                        self.signature_agent.method_names.index(method_name)
                    ] += 1
                    shutil.rmtree(implementation_results_path)
                    api_info["error"] = "Implementation failed"
                    return api_info
            else:
                # If the error is not due to a missing method, increment the error count
                self.error_counts[
                    self.signature_agent.method_names.index(method_name)
                ] += 1
                shutil.rmtree(implementation_results_path)
                api_info["error"] = stacktrace
                return api_info
        # If the implementation is correct, mark it as implemented
        self.implemented[self.signature_agent.method_names.index(method_name)] = True
        api_info["error"] = None
        return api_info

    def _trace_execution(self, frame, event, arg):
        if event == "line":
            filename = frame.f_globals.get("__file__", None)
            if filename:
                lineno = frame.f_lineno
                line = linecache.getline(filename, lineno).strip()
                if lineno > self.namespace_line:
                    return self._trace_execution
                if "import math" in line:
                    return self._trace_execution
                if "import" in line:
                    self.namespace_line = lineno
                    return self._trace_execution
                with open(self.trace_file_path, "a+") as f:
                    f.write(f"<p>{lineno}: {line}</p>\n")
        return self._trace_execution

    def _execute_file(self, program_executable_path):
        sys.settrace(self._trace_execution)
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(30)
            runpy.run_path(program_executable_path, init_globals=self.namespace)
            signal.alarm(0)
        except TimeoutException as e:
            stacktrace = traceback.format_exc()
            return e, stacktrace
        except Exception as e:
            stacktrace = traceback.format_exc()
            return e, stacktrace
        finally:
            sys.settrace(None)
        return

    def _get_docstring_types(self, docstring):
        args_pattern = re.compile(r"Args:\s*((?:\s+\w+ \(\w+\): .+\n)+)")
        args_match = args_pattern.search(docstring)
        args_section = args_match.group(1) if args_match else ""

        returns_pattern = re.compile(r"Returns:\s+(\w+): .+")
        returns_match = returns_pattern.search(docstring)
        returns_section = returns_match.group(1) if returns_match else ""

        arg_types = re.findall(r"\s+(\w+) \((\w+)\):", args_section)
        return arg_types, returns_section

    def _get_return_code(self, returns):
        if self.dataset in ["clevr", "gqa"]:
            if returns == "string":
                return '\n\treturn ""'
            elif returns == "image":
                return "\n\treturn image"
            elif returns == "int":
                return "\n\treturn 25"
            elif returns == "float":
                return "\n\treturn 1.0"
            elif returns == "list":
                return "\n\treturn [[25, 25]]"
            elif returns == "bool":
                return "\n\treturn False"
            elif returns == "tuple":
                return "\n\treturn (50, 50)"
            else:
                return "\n\treturn 1"
        else:
            if returns == "string":
                return '\n\treturn ""'
            elif returns == "image":
                return "\n\treturn image"
            elif returns == "int":
                return "\n\treturn 25"
            elif returns == "float":
                return "\n\treturn 1.0"
            elif returns == "list":
                return "\n\treturn [[25, 25, 50, 50]]"
            elif returns == "bool":
                return "\n\treturn False"
            elif returns == "tuple":
                return "\n\treturn (50, 50)"
            else:
                return "\n\treturn 1"

    def get_api_implementations(self, results_folder_path, prompt=None):
        # Use dataset-specific prompt if none provided
        if prompt is None:
            if self.dataset == "clevr":
                prompt = API_PROMPT_CLEVR
            elif self.dataset == "gqa":
                prompt = API_PROMPT_GQA
            else:
                prompt = API_PROMPT

        headers = self.signature_agent.headers

        folder_name = "api_generator"
        results_folder_path = os.path.join(
            results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path)

        file_path = os.path.join(results_folder_path, "api_implementation.html")

        for header in headers:
            implementation, output = self(
                header["method_name"],
                header["docstring"],
                header["signature"],
                results_folder_path,
                prompt=prompt,
            )

            if self.write_results:

                method_name = (
                    re.compile(r"def (\w+)\s*\(.*\):").search(header["signature"]).group(1)
                )
                implementation_results_path = os.path.join(
                    results_folder_path, f"{method_name}"
                )
                if not os.path.exists(implementation_results_path):
                    continue
                with open(os.path.join(implementation_results_path, f"{method_name}.html"), "wb+") as file:

                    generated_signatures = [
                        header["docstring"] + header["signature"]
                        for header in self.signature_agent.headers
                        if header["method_name"] != header["method_name"]
                    ]
                    generated_signatures = "\n\n".join(generated_signatures)

                    file.write((f"<h1>Signature</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{header['docstring'] + header['signature']}</code>\n".replace(
                                "\n", "<br>"
                            )
                        ).encode("utf-8")
                    )

                    file.write((f"<h1>Prompt</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{prompt.format(predef_signatures=self.signature_agent.predef_signatures, generated_signatures=generated_signatures, docstring=header['docstring'], signature=header['signature'])}</code>\n".replace(
                                "\n", "<br>"
                            )
                        ).encode("utf-8")
                    )

                    file.write((f"<h1>LLM Output</h1>\n").encode("utf-8"))
                    file.write(
                        (f"<code>{output}</code>\n".replace("\n", "<br>")).encode(
                            "utf-8"
                        )
                    )

                    file.write((f"<h1>Implementation</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{implementation}</code>\n".replace("\n", "<br>")
                        ).encode("utf-8")
                    )
                    file.close()
            self.pbar.update(1)
        self.pbar.close()

        api_path = os.path.join(results_folder_path, "api.json")

        with open(api_path, "w+") as file:
            json.dump(self.api, file)

        return api_path, self.api


class ProgramAgent(Agent):
    def __init__(
        self, api_agent, model_name="gpt-4o", write_results=True, dataset="clevr"
    ):
        super().__init__(model_name, write_results, dataset=dataset)
        self.api_agent = api_agent
        self.programs = []

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def __call__(self, question, prompt):

        api_text = "\n".join(
            [api["docstring"] + api["signature"] for api in self.api_agent.api]
        )

        prompt = prompt.format(
            predef_signatures=self.api_agent.signature_agent.predef_signatures,
            api=api_text,
            question=question["question"],
        )

        output, messages = self.generator.generate(prompt)
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        program = re.findall(r"<program>(.*?)</program>", output, re.DOTALL)
        self.programs.append(
            {
                "image_index": question["image_index"],
                "question_index": question["question_index"],
                "program": program,
                "prompt": prompt,
                "output": output,
                "messages": messages,
                "model_name": self.generator.model_name,
            }
        )

        return program, output

    def get_programs(
        self,
        questions_data,
        images_folder_path,
        results_folder_path,
        prompt=None,
    ):
        # Use dataset-specific prompt if none provided
        if prompt is None:
            if self.dataset == "clevr":
                prompt = PROGRAM_PROMPT_CLEVR
            elif self.dataset == "gqa":
                prompt = PROGRAM_PROMPT_GQA
            else:
                prompt = PROGRAM_PROMPT

        folder_name = "program_generator"
        results_folder_path = os.path.join(
            results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path)

        for question_data in tqdm(questions_data):
            html_path = os.path.join(
                results_folder_path,
                f"image_{question_data['image_index']}_question_{question_data['question_index']}.html",
            )

            question = question_data["question"]
            program, output = self(question_data, prompt)

            if self.write_results:
                with open(html_path, "wb+") as file:

                    # get image
                    image = Image.open(
                        os.path.join(
                            images_folder_path, question_data["image_filename"]
                        )
                    )
                    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    rgb_image = image.convert("RGB")
                    image_io = io.BytesIO()
                    rgb_image.save(image_io, format="PNG")
                    image_bytes = base64.b64encode(image_io.getvalue()).decode("ascii")

                    # Write question and image
                    file.write((f"<h1>{question}</h1>\n").encode("utf-8"))
                    file.write(
                        (f"<img src='data:image/jpeg;base64,{image_bytes}'>\n").encode(
                            "utf-8"
                        )
                    )

                    api_text = "\n".join(
                        [
                            api["docstring"] + api["signature"]
                            for api in self.api_agent.api
                        ]
                    )

                    file.write((f"<h1>Prompt</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{prompt.format(predef_signatures=self.api_agent.signature_agent.predef_signatures, api=api_text, question=question_data['question'])}</code>\n".replace(
                                "\n", "<br>"
                            )
                        ).encode("utf-8")
                    )

                    file.write((f"<h1>LLM Outputs</h1>\n").encode("utf-8"))
                    if isinstance(output, list):
                        for out in output:
                            file.write(
                                (f"<code>{out}</code>\n".replace("\n", "<br>")).encode(
                                    "utf-8"
                                )
                            )
                    else:
                        file.write(
                            (f"<code>{output}</code>\n".replace("\n", "<br>")).encode(
                                "utf-8"
                            )
                        )

                    file.write((f"<h1>Program</h1>\n").encode("utf-8"))
                    if len(program) > 0:
                        file.write(
                            (f"<code>{program[0]}</code>\n".replace("\n", "<br>")).encode(
                                "utf-8"
                            )
                        )
                    else:
                        file.write((f"<p>No program found</p>").encode("utf-8"))

                    file.close()

        programs_path = os.path.join(results_folder_path, "programs.json")

        with open(programs_path, "w+") as file:
            json.dump(self.programs, file)

        return programs_path, self.programs

