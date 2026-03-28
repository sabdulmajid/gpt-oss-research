import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_oss_research.cli.run_model_eval import main


if __name__ == "__main__":
    main()

