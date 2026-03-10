import os
import subprocess
import json
import time
from google import genai
from dotenv import load_dotenv

# 1. Configuration
load_dotenv() # This loads the secrets from your .env file
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY is not set in the .env file.")
    exit()

client = genai.Client(api_key=api_key)

# 2. File Paths -- Update the File Paths section of the code to use generic placeholders instead of hardcoding
app_dir = r"C:\Users\Aman.kumar\Downloads\Communication-Service-development\Communication-Service-development\app" 
# for test purpose

input_filepath = os.path.join(app_dir, r"controllers\auth.controller.js")
output_filepath = os.path.join(app_dir, r"tests\controllers\auth.controller.test.js")

coverage_dir = os.path.join(app_dir, "coverage")
coverage_file = os.path.join(coverage_dir, "coverage-summary.json")

try:
    with open(input_filepath, "r", encoding="utf-8") as file:
        js_code = file.read()
except FileNotFoundError:
    print(f"Could not find the JS file at: {input_filepath}")
    exit()

def generate_tests_with_ai(prompt):
    max_api_retries = 3
    for attempt in range(max_api_retries):
        try:
            print("Asking Gemini to write/fix tests...")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            code = response.text
            if code.startswith("```"):
                code = "\n".join(code.split("\n")[1:-1])
            if code.startswith("javascript"):
                code = code.replace("javascript\n", "", 1)
            return code.strip()
            
        except Exception as e:
            print(f"\n[API Error]: {e}")
            if attempt < max_api_retries - 1:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
            else:
                print("Failed to reach Gemini API. Exiting.")
                exit()

def run_tests_and_get_coverage():
    print("Running Jest and calculating coverage...")
    
    relative_test_path = "tests/controllers/auth.controller.test.js"
    
    command = [
        "npx", "jest", relative_test_path,
        "--coverage",
        "--coverageReporters=\"text\"",
        "--coverageReporters=\"json-summary\"",
        "--coverageDirectory=\"coverage\""
    ]
    
    # FIX: Added encoding="utf-8" and errors="replace" so Jest's checkmarks don't crash Windows
    result = subprocess.run(
        " ".join(command),
        cwd=app_dir,
        capture_output=True,
        text=True,
        encoding="utf-8", 
        errors="replace",
        shell=True
    )
    
    # FIX: Safely combine outputs in case one of them is empty (None)
    stdout_str = result.stdout if result.stdout else ""
    stderr_str = result.stderr if result.stderr else ""
    terminal_output = stdout_str + "\n" + stderr_str
    
    tests_passed = result.returncode == 0
    
    coverage = 0.0
    if os.path.exists(coverage_file):
        try:
            with open(coverage_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
                coverage = summary.get("total", {}).get("lines", {}).get("pct", 0.0)
        except Exception as e:
            print(f"Could not parse JSON coverage: {e}")
            
    return tests_passed, terminal_output, coverage

# 3. The Node.js / Jest Prompt
current_prompt = f"""
You are an expert Node.js developer. Write unit tests for the following JavaScript file using Jest.
Follow these rules strictly:
1. Use `jest.mock()` to mock any external modules, database models, or services imported by this file.
2. Cover the happy path and error handling (try/catch blocks).
3. Ensure you import the functions/classes correctly based on how they are exported.
4. Return ONLY the raw JavaScript code for the test file. Do not include markdown blocks (```javascript), explanations, or any other text.

Here is the source code to test:
{js_code}
"""

# 4. The Self-Healing Loop
max_retries = 5
target_coverage = 80.0

for attempt in range(1, max_retries + 1):
    print(f"\n--- Attempt {attempt} of {max_retries} ---")
    
    generated_tests = generate_tests_with_ai(current_prompt)
    with open(output_filepath, "w", encoding="utf-8") as file:
        file.write(generated_tests)
        
    tests_passed, terminal_output, coverage = run_tests_and_get_coverage()
    
    print(f"Tests Passed: {tests_passed}")
    print(f"Code Coverage: {coverage}%")
    
    print("\n--- Jest Terminal Output ---")
    print(terminal_output)
    print("----------------------------\n")
    
    if tests_passed and coverage >= target_coverage:
        print("\nSUCCESS! The AI wrote passing tests with great coverage.")
        break
        
    if attempt == max_retries:
        print("\nReached maximum retries. Please review the tests manually.")
        break
        
    print("Tests failed or coverage was below 80%. Sending the error report back to Gemini...")
    current_prompt = f"""
    You previously generated these Jest tests:
    {generated_tests}
    
    When I ran them using `npx jest`, here was the terminal output and error report:
    {terminal_output}
    
    The current line coverage is only {coverage}%.
    
    Please fix the failing tests. If tests pass but coverage is low, add more test cases to reach at least {target_coverage}% coverage.
    Pay close attention to any "Cannot find module" errors or mock implementation issues.
    Return ONLY the raw JavaScript code. Do not include markdown blocks or text.
    
    Original source code being tested:
    {js_code}
    """