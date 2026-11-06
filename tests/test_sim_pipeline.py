"""The simulator's build pipeline: its stages are wired in order, without running any of them;
what the simulator loads is committed."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from drivingbench.sim.course import BOUNDS, CONES, COURSE, GROUND, REPO, SITE, SITE_FILE
from drivingbench.sim.pipeline.__main__ import main, stages


def paths(command):
    return [Path(part) for part in command if isinstance(part, Path)]


def test_every_stage_reads_only_what_earlier_stages_made():
    plan = stages(SITE)
    made = {}
    for s in plan:
        for output in s.outputs:
            assert output not in made, f"{s.name} and {made[output]} both make {output}"
            made[output] = s.name
    for k, s in enumerate(plan):
        earlier = {p for t in plan[:k] for p in t.outputs}
        later = {p: t.name for t in plan[k + 1 :] for p in t.outputs}
        for command in s.commands:
            for p in paths(command):
                assert p not in later, f"{s.name} reads {p}, which {later[p]} makes after it"
                if p in made and made[p] != s.name:
                    assert p in earlier
            module = command[2] if command[1:2] == ["-m"] else None
            assert module is None or importlib.util.find_spec(module), module


def test_the_listing_shows_every_stage_and_its_commands(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build"])
    main()
    shown = capsys.readouterr().out
    assert SITE["name"] in shown
    for s in stages(SITE):
        assert f"  {s.name}: {s.makes}" in shown
    work, day = SITE["work"], SITE["fit_day"]
    assert f"python -m drivingbench.sim.pipeline.layout {work}/verified.json" in shown
    assert f"python -m drivingbench.sim.fit --series runs/sim/series --day {day}" in shown
    assert str(REPO) not in shown  # commands read as typed from the repository root


def test_another_site_file_moves_the_work_and_outputs(tmp_path, monkeypatch, capsys):
    site = json.loads(SITE_FILE.read_text()) | {
        "name": "Elsewhere",
        "work": "runs/sim/build/elsewhere",
    }
    (tmp_path / "site.json").write_text(json.dumps(site))
    monkeypatch.setattr(sys, "argv", ["build", "--site", str(tmp_path / "site.json")])
    main()
    shown = capsys.readouterr().out
    assert shown.startswith("Elsewhere")
    assert "python -m drivingbench.sim.pipeline.reconstruct runs/sim/build/elsewhere" in shown
    monkeypatch.setattr(sys, "argv", ["build", "--run", "nonsense"])
    with pytest.raises(SystemExit):
        main()


def test_what_the_simulator_loads_is_committed_with_the_site():
    for path in (COURSE, GROUND, CONES, BOUNDS):
        assert path.is_relative_to(REPO / "sim") and path.is_file(), path
    assert all(not p.startswith("sim/") for p in SITE["splats"].values())  # built, not committed
