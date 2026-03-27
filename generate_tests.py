"""
Self-Healing Jest Test Generator
=================================
Usage:
    python generate_tests.py <project_dir> [options]

Options:
    --src-dir DIR       Source sub-directory inside project (auto-detected if omitted)
    --test-dir DIR      Where to write tests, relative to src-dir (default: tests)
    --coverage N        Minimum line-coverage % to consider a file done (default: 80)
    --retries N         Max AI fix attempts per file before giving up (default: 5)
    --model MODEL       Gemini model to use (default: gemini-2.5-flash)
    --extensions        Comma-separated extensions to process (default: .js,.ts)

Environment:
    No API key required — Ollama runs fully locally!

Example:
    python generate_tests.py /path/to/my-node-app --src-dir app --coverage 85
"""

import os
import sys
import re
import json
import time
import argparse
import subprocess
import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Tee — write to both stdout and a log file simultaneously
# ---------------------------------------------------------------------------

class Tee:
    """Redirect sys.stdout so every print() goes to console AND a log file."""

    def __init__(self, log_path: str):
        self._console = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
        print(f"  📋  Logging all output to: {log_path}", file=self._console)

    def write(self, data: str):
        self._console.write(data)
        self._console.flush()
        self._log.write(data)

    def flush(self):
        self._console.flush()
        self._log.flush()

    def close(self):
        self._log.close()
        sys.stdout = self._console

# ---------------------------------------------------------------------------
# Optional deps — give a clear error if missing
# ---------------------------------------------------------------------------
try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not installed. Run: pip install requests")
    sys.exit(1)




# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directories to never walk into when discovering source files
SKIP_DIRS: set[str] = {
    "node_modules", "coverage", ".git", "dist", "build", "out",
    ".next", ".nuxt", "vendor", "__pycache__", ".cache",
}

# File-name fragments that indicate the file IS already a test — skip it
SKIP_FILE_FRAGMENTS: list[str] = [".test.", ".spec.", ".e2e."]

# Config-like files we don't want to generate tests for
SKIP_FILE_NAMES: set[str] = {
    "jest.config.js", "jest.config.ts",
    "babel.config.js", "babel.config.ts",
    "webpack.config.js", "rollup.config.js",
    ".eslintrc.js",
}

# Common candidate names for the source root inside a Node project
SRC_DIR_CANDIDATES: list[str] = ["src", "app", "lib", "server", "."]


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-healing Jest test generator powered by Ollama (local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "project_dir",
        help="Absolute or relative path to the Node.js project root",
    )
    parser.add_argument(
        "--src-dir",
        default=None,
        help="Source sub-directory inside the project (auto-detected if omitted)",
    )
    parser.add_argument(
        "--test-dir",
        default="tests",
        help="Directory for generated tests, relative to src-dir (default: tests)",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=80.0,
        help="Minimum line-coverage %% to consider a file done (default: 80)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Max AI fix attempts per file before giving up (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:14b",
        help="Ollama model name (default: qwen2.5-coder:14b)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--extensions",
        default=".js,.ts",
        help="Comma-separated file extensions to process (default: .js,.ts)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def check_ollama_running(ollama_url: str, model: str) -> None:
    """Verify Ollama is running and the requested model is available."""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        # Normalize: ollama tags may include ":latest" suffix
        available_base = [m.split(":")[0] for m in available]
        model_base = model.split(":")[0]
        if model not in available and model_base not in available_base:
            print(f"\nERROR: Model '{model}' is not pulled in Ollama.")
            print(f"  Available models: {available or '(none)'}")
            print(f"  Run: ollama pull {model}\n")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Ollama is not running at {ollama_url}")
        print("  Start it with: ollama serve")
        print("  Or download from: https://ollama.com\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: Could not reach Ollama: {exc}\n")
        sys.exit(1)


def auto_detect_src_dir(project_dir: str) -> str:
    """Return the first existing candidate directory, or raise."""
    for candidate in SRC_DIR_CANDIDATES:
        path = os.path.join(project_dir, candidate)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not auto-detect a source directory in '{project_dir}'. "
        "Use --src-dir to specify one explicitly."
    )


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def discover_source_files(src_dir: str, extensions: list[str]) -> list[str]:
    """
    Walk src_dir and return all source files worth testing.
    Excludes test files, config files, and unwanted directories.
    """
    found: list[str] = []
    ext_set = set(extensions)

    for root, dirs, files in os.walk(src_dir, topdown=True):
        # Prune unwanted directories in-place so os.walk won't recurse into them
        dirs[:] = sorted(
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        )

        for filename in sorted(files):
            _, ext = os.path.splitext(filename)
            if ext not in ext_set:
                continue
            if filename in SKIP_FILE_NAMES:
                continue
            if any(frag in filename for frag in SKIP_FILE_FRAGMENTS):
                continue
            found.append(os.path.join(root, filename))

    return found


# ---------------------------------------------------------------------------
# AI Interaction
# ---------------------------------------------------------------------------

def call_ollama(ollama_url: str, model: str, prompt: str, api_retries: int = 3) -> str | None:
    """
    Call the local Ollama API and return clean JavaScript code.
    No rate limits, no quota — runs fully on your machine.
    Returns None if all retries fail (e.g. Ollama crashed mid-run).
    """
    endpoint = f"{ollama_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,      # low temp = more deterministic code
            "num_predict": 8192,     # max tokens to generate
        }
    }
    for attempt in range(1, api_retries + 1):
        try:
            print(f"    🤖 Calling Ollama ({model})...", flush=True)
            resp = requests.post(endpoint, json=payload, timeout=300)  # 5 min timeout
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return strip_markdown_fences(raw)
        except requests.exceptions.Timeout:
            print(f"    [Timeout on attempt {attempt}/{api_retries}] — model is taking too long")
        except requests.exceptions.ConnectionError:
            print(f"    [Connection error on attempt {attempt}/{api_retries}] — is Ollama still running?")
        except Exception as exc:
            print(f"    [Error on attempt {attempt}/{api_retries}]: {exc}")
        if attempt < api_retries:
            print("    Waiting 5 s before retry...")
            time.sleep(5)
    print("    ❌ Ollama unreachable after all retries.")
    return None


def strip_markdown_fences(text: str) -> str:
    """Remove opening and closing triple-backtick fences from AI responses."""
    text = text.strip()
    # Remove opening fence: ```javascript, ```js, ```typescript, ```ts, or just ```
    text = re.sub(r"^```(?:javascript|js|typescript|ts|jsx|tsx)?\s*\n", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Test Execution
# ---------------------------------------------------------------------------

def run_jest(
    project_dir: str,
    test_file_rel: str,
    coverage_out_dir: str,
) -> tuple[bool, str, float]:
    """
    Run Jest for a single test file and return (tests_passed, terminal_output, coverage_pct).
    coverage_pct is extracted from the JSON summary; falls back to parsing text output.
    """
    # Normalise path separators for Jest (always use forward slashes)
    jest_test_path = test_file_rel.replace("\\", "/")

    # On Windows, npx is npx.cmd and cannot be found without shell=True
    is_windows = sys.platform == "win32"
    npx = "npx"   # shell=True on Windows resolves .cmd automatically

    cmd = [
        npx, "jest",
        jest_test_path,
        "--coverage",
        "--coverageReporters=text",
        "--coverageReporters=json-summary",
        f"--coverageDirectory={coverage_out_dir}",
        "--forceExit",
        "--no-cache",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=is_windows,   # Required on Windows for .cmd executables like npx
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    terminal_output = stdout + "\n" + stderr
    tests_passed = result.returncode == 0

    coverage_pct = _read_coverage_json(project_dir, coverage_out_dir)
    if coverage_pct == 0.0:
        # Fallback: parse Jest's text table from the terminal output
        coverage_pct = _parse_coverage_from_text(terminal_output)

    return tests_passed, terminal_output, coverage_pct


def _read_coverage_json(project_dir: str, coverage_out_dir: str) -> float:
    """
    Parse coverage-summary.json produced by Jest --coverageReporters=json-summary.
    Returns the total line-coverage percentage, or 0.0 if unavailable.
    """
    summary_path = os.path.join(project_dir, coverage_out_dir, "coverage-summary.json")
    if not os.path.exists(summary_path):
        return 0.0
    try:
        with open(summary_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return float(data.get("total", {}).get("lines", {}).get("pct", 0.0))
    except Exception as exc:
        print(f"    ⚠  Could not parse coverage JSON: {exc}")
        return 0.0


def _parse_coverage_from_text(output: str) -> float:
    """
    Extract line-coverage percentage from Jest's text coverage table.
    Jest prints lines like:
        All files   |   82.14 |   75.00 |   83.33 |   82.14 |
        Lines      :  82.14% ( 46/56 )
    """
    # Format 1: table row — "All files | 82.14 | ..."
    m = re.search(r"All files\s*\|\s*([\d.]+)", output)
    if m:
        return float(m.group(1))

    # Format 2: summary line — "Lines : 82.14% ( 46/56 )"
    m = re.search(r"Lines\s*[:|]\s*([\d.]+)\s*%", output, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return 0.0


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def build_initial_prompt(source_code: str, source_rel: str) -> str:
    return f"""You are an expert Node.js developer. Write complete Jest unit tests for the file below.

Rules — follow all of them:
1. Use `jest.mock()` to mock every external module, database model, or service imported by this file.
2. Cover the happy path AND all error / edge cases (especially try/catch blocks).
3. Import the functions / classes exactly as they are exported (named vs default export).
4. Do NOT import the real database or any real network dependency — mock everything.
5. Return ONLY the raw JavaScript code for the test file.
   No markdown fences (```), no explanations, no comments outside the test code itself.

File being tested: {source_rel}

Source code:
{source_code}
"""


def build_fix_prompt(
    source_code: str,
    source_rel: str,
    generated_tests: str,
    terminal_output: str,
    coverage_pct: float,
    target_coverage: float,
    tests_passed: bool,
) -> str:
    if not tests_passed:
        issue = "The tests failed (non-zero exit code from Jest). Fix ALL failing tests."
    else:
        issue = (
            f"The tests passed but line coverage is only {coverage_pct:.1f}% "
            f"(target: {target_coverage:.1f}%). "
            "Add more test cases to cover uncovered branches, error paths, and edge cases."
        )

    return f"""You previously generated these Jest tests for '{source_rel}':

--- GENERATED TESTS ---
{generated_tests}
--- END ---

After running `npx jest`, the terminal output was:

--- JEST OUTPUT ---
{terminal_output.strip()}
--- END ---

Problem: {issue}

Instructions:
- Fix every failing test or add the missing test cases.
- Pay close attention to "Cannot find module", mock errors, and assertion failures.
- Do NOT change the file being tested — only the test file.
- Return ONLY the raw JavaScript code for the corrected test file.
  No markdown fences, no explanations.

Original source code (do NOT modify):
{source_code}
"""


# ---------------------------------------------------------------------------
# Per-File Processing
# ---------------------------------------------------------------------------

def derive_test_path(
    src_file: str,
    src_dir: str,
    test_dir: str,
) -> tuple[str, str]:
    """
    Given a source file path, return:
      - test_abs: absolute path where the test file should be written
      - test_rel_to_src: path relative to src_dir (used for Jest's test path argument)

    Example:
        src_file  = /project/app/controllers/auth.controller.js
        src_dir   = /project/app
        test_dir  = tests
        →
        test_abs        = /project/app/tests/controllers/auth.controller.test.js
        test_rel_to_src = tests/controllers/auth.controller.test.js
    """
    rel = os.path.relpath(src_file, src_dir)             # controllers/auth.controller.js
    base, ext = os.path.splitext(rel)                     # controllers/auth.controller | .js
    test_rel_to_src = os.path.join(test_dir, base + f".test{ext}")  # tests/controllers/auth.controller.test.js
    test_abs = os.path.join(src_dir, test_rel_to_src)
    return test_abs, test_rel_to_src


def process_file(
    src_file: str,
    project_dir: str,
    src_dir: str,
    test_dir: str,
    coverage_out_dir: str,
    ollama_url: str,
    model: str,
    target_coverage: float,
    max_retries: int,
) -> tuple[bool, float]:
    """
    Full self-healing loop for one source file.
    Returns (success, final_coverage_pct).
    """
    source_rel = os.path.relpath(src_file, project_dir)

    # Read the source file
    try:
        with open(src_file, "r", encoding="utf-8") as fh:
            source_code = fh.read()
    except OSError as exc:
        print(f"  ❌ Cannot read source file: {exc}")
        return False, 0.0

    if not source_code.strip():
        print("  ⚠  File is empty — skipping.")
        return False, 0.0

    # Derive where the test file lives
    test_abs, test_rel_to_src = derive_test_path(src_file, src_dir, test_dir)
    # Jest is run from project_dir, so we need the path relative to project_dir
    test_rel_to_project = os.path.relpath(test_abs, project_dir)

    os.makedirs(os.path.dirname(test_abs), exist_ok=True)

    prompt = build_initial_prompt(source_code, source_rel)
    generated_tests: str = ""
    final_coverage: float = 0.0

    for attempt in range(1, max_retries + 1):
        print(f"\n  ─── Attempt {attempt}/{max_retries} ───────────────────────", flush=True)

        # ── Generate ──────────────────────────────────────────────────────
        generated_tests = call_ollama(ollama_url, model, prompt)
        if generated_tests is None:
            print("  ❌ Skipping this attempt — AI call failed.")
            if attempt == max_retries:
                print("  ⛔ Max retries reached. Please write tests manually.")
                return False, final_coverage
            continue

        # ── Write test file ───────────────────────────────────────────────
        with open(test_abs, "w", encoding="utf-8") as fh:
            fh.write(generated_tests)
        print(f"    📝 Test file written → {test_rel_to_project}")

        # ── Run Jest ──────────────────────────────────────────────────────
        tests_passed, terminal_output, coverage_pct = run_jest(
            project_dir, test_rel_to_project, coverage_out_dir
        )
        final_coverage = coverage_pct

        # ── Report ────────────────────────────────────────────────────────
        pass_icon = "✅" if tests_passed else "❌"
        cov_icon  = "✅" if coverage_pct >= target_coverage else "⚠️ "
        print(f"    {pass_icon} Tests: {'PASSED' if tests_passed else 'FAILED'}")
        print(f"    {cov_icon}  Coverage: {coverage_pct:.1f}% (target ≥ {target_coverage:.1f}%)")

        # ── Success check ─────────────────────────────────────────────────
        if tests_passed and coverage_pct >= target_coverage:
            print(f"\n  🎉 Done! {coverage_pct:.1f}% coverage achieved.")
            return True, coverage_pct

        # ── Max retries? ──────────────────────────────────────────────────
        if attempt == max_retries:
            print("\n  ⛔ Max retries reached. Please review tests manually.")
            _print_jest_output(terminal_output)
            return False, coverage_pct

        # ── Build fix prompt and retry ────────────────────────────────────
        print("  🔁 Sending error report back to Ollama for a fix...")
        _print_jest_output(terminal_output)
        prompt = build_fix_prompt(
            source_code, source_rel,
            generated_tests, terminal_output,
            coverage_pct, target_coverage,
            tests_passed,
        )

    return False, final_coverage


def _print_jest_output(output: str, max_lines: int = 60) -> None:
    """Print Jest terminal output, truncated if very long."""
    lines = output.strip().splitlines()
    if not lines:
        return
    print("\n    ┌─ Jest Output " + "─" * 46)
    for line in lines[:max_lines]:
        print("    │ " + line)
    if len(lines) > max_lines:
        print(f"    │ … ({len(lines) - max_lines} more lines hidden)")
    print("    └" + "─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Resolve project directory ─────────────────────────────────────────
    project_dir = os.path.abspath(args.project_dir)
    if not os.path.isdir(project_dir):
        print(f"ERROR: Project directory not found: {project_dir}")
        sys.exit(1)

    # ── Load API key ──────────────────────────────────────────────────────
    # ── Start logging — tee all output to a timestamped log file ─────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(project_dir, f"test_gen_{timestamp}.log")
    tee = Tee(log_path)
    sys.stdout = tee

    ollama_url = args.ollama_url.rstrip("/")
    check_ollama_running(ollama_url, args.model)

    # ── Resolve source directory ──────────────────────────────────────────
    if args.src_dir:
        src_dir = os.path.join(project_dir, args.src_dir)
        if not os.path.isdir(src_dir):
            print(f"ERROR: --src-dir '{args.src_dir}' does not exist in {project_dir}")
            sys.exit(1)
    else:
        try:
            src_dir = auto_detect_src_dir(project_dir)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    test_dir       = args.test_dir
    target_coverage = args.coverage
    max_retries    = args.retries
    model          = args.model
    ollama_url     = args.ollama_url.rstrip("/")
    extensions     = [e.strip() for e in args.extensions.split(",")]
    coverage_out_dir = "coverage"   # always inside project_dir; Jest default

    # ── Banner ────────────────────────────────────────────────────────────
    src_rel  = os.path.relpath(src_dir, project_dir)
    test_abs = os.path.join(src_dir, test_dir)

    print()
    print("═" * 62)
    print("  🚀  Self-Healing Jest Test Generator")
    print("═" * 62)
    print(f"  Project dir   : {project_dir}")
    print(f"  Source dir    : {src_rel}")
    print(f"  Test output   : {os.path.relpath(test_abs, project_dir)}")
    print(f"  Coverage dir  : {coverage_out_dir}")
    print(f"  Min coverage  : {target_coverage:.0f}%")
    print(f"  Max retries   : {max_retries}")
    print(f"  Ollama model  : {model}")
    print(f"  Extensions    : {', '.join(extensions)}")
    print("═" * 62)

    # ── Discover source files ─────────────────────────────────────────────
    src_files = discover_source_files(src_dir, extensions)

    # Filter out anything already inside the test_dir to be safe
    src_files = [
        f for f in src_files
        if not os.path.relpath(f, src_dir).startswith(test_dir + os.sep)
        and not os.path.relpath(f, src_dir).startswith(test_dir + "/")
    ]

    if not src_files:
        print(f"\n  ⚠  No {'/'.join(extensions)} source files found under '{src_rel}'.")
        print("     Use --src-dir or --extensions to adjust the search.\n")
        sys.exit(0)

    print(f"\n  📂  Found {len(src_files)} source file(s) to process:\n")
    for f in src_files:
        print(f"       • {os.path.relpath(f, project_dir)}")
    print()

    # ── Process each file ─────────────────────────────────────────────────
    results: dict[str, dict] = {}

    for idx, src_file in enumerate(src_files, start=1):
        rel = os.path.relpath(src_file, project_dir)
        print()
        print("═" * 62)
        print(f"  📄  [{idx}/{len(src_files)}]  {rel}")
        print("═" * 62)

        success, coverage_pct = process_file(
            src_file     = src_file,
            project_dir  = project_dir,
            src_dir      = src_dir,
            test_dir     = test_dir,
            coverage_out_dir = coverage_out_dir,
            ollama_url   = ollama_url,
            model        = model,
            target_coverage = target_coverage,
            max_retries  = max_retries,
        )

        results[rel] = {"success": success, "coverage": coverage_pct}

    # ── Final summary ─────────────────────────────────────────────────────
    passed_files = [(f, r) for f, r in results.items() if r["success"]]
    failed_files = [(f, r) for f, r in results.items() if not r["success"]]

    total_coverage = (
        sum(r["coverage"] for r in results.values()) / len(results)
        if results else 0.0
    )

    print()
    print("═" * 62)
    print("  📊  FINAL SUMMARY")
    print("═" * 62)
    print(f"  Total files processed : {len(results)}")
    print(f"  ✅  Passed            : {len(passed_files)}")
    print(f"  ❌  Need manual review: {len(failed_files)}")

    print(f"  📈  Avg line coverage : {total_coverage:.1f}%")
    print()

    if passed_files:
        print("  ✅  Automated successfully:")
        for f, r in passed_files:
            bar = _coverage_bar(r["coverage"])
            print(f"       {bar}  {r['coverage']:5.1f}%  {f}")

    if failed_files:
        print()
        print("  ❌  Need manual attention:")
        for f, r in failed_files:
            bar = _coverage_bar(r["coverage"])
            print(f"       {bar}  {r['coverage']:5.1f}%  {f}")



    print()
    print("═" * 62)
    print(f"  📋  Full log saved to: {log_path}")
    print()

    tee.close()


def _coverage_bar(pct: float, width: int = 10) -> str:
    """Render a tiny ASCII progress bar, e.g. [████░░░░░░]"""
    filled = round(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


if __name__ == "__main__":
    main()
