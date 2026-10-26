from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from q_attention.experiments.srm_handoff import main
if __name__=='__main__':raise SystemExit(main())
