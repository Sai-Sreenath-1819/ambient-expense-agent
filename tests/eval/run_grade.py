import sys
import os
import google.auth
from google.auth.credentials import AnonymousCredentials

# Mock google.auth.default to prevent DefaultCredentialsError
def mock_default(*args, **kwargs):
    return AnonymousCredentials(), "dummy-project"

google.auth.default = mock_default

from google.agents.cli.eval.cmd_grade import cmd_grade

if __name__ == "__main__":
    sys.argv = [
        "grade",
        "--traces", "artifacts/traces/generated_traces.json",
        "--config", "tests/eval/eval_config.yaml"
    ]
    os.environ["GOOGLE_CLOUD_PROJECT"] = "dummy-project"
    cmd_grade.main()
