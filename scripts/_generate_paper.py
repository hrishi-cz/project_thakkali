"""Helper: generate paper.md and paper.tex from current diary/results/ data."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from research.paper_generator import PaperGenerator
from research.experiment_collector import ExperimentCollector
from research.ablation import build_ablation

exps = ExperimentCollector().collect()
abl = build_ablation(exps)
gen = PaperGenerator(exps, abl)

md = gen.generate_full_paper()
_md_path = _ROOT / "diary" / "results" / "paper.md"
_md_path.parent.mkdir(parents=True, exist_ok=True)
_md_path.write_text(md, encoding="utf-8")
print(f"Paper markdown written to {_md_path} ({len(md)} chars)")

try:
    tex = gen.generate_latex()
    _tex_path = _ROOT / "diary" / "results" / "paper.tex"
    _tex_path.write_text(tex, encoding="utf-8")
    print(f"LaTeX written to {_tex_path} ({len(tex)} chars)")
except Exception as e:
    print(f"LaTeX generation skipped: {e}")
