"""Railway entry point -- runs training data generation."""
import subprocess
import sys

sys.exit(subprocess.call([
    sys.executable, "scripts/generate_training_data.py",
    "--output", "/app/training_data.jsonl",
    "--easy-count", "7000",
    "--hard-count", "3000",
    "--workers", "32",
    "--easy-timeout-s", "60",
    "--hard-timeout-s", "240",
    "--progress",
]))
