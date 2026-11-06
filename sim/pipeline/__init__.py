"""The simulator's build: from a site's drives to everything the simulator loads but the course.

`python -m drivingbench.sim.pipeline` lists the stages and runs them (sim/pipeline/__main__.py);
every stage but Brush's training is a module here (or in sim/ and sim/explorer/) that also runs on
its own."""
