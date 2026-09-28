"""V2 entry point with an explicitly fixed no-acceleration evaluation control."""
import sys
import sdaea_streaming_v2 as v2

idle = "--idle-control" in sys.argv
if idle:
    sys.argv.remove("--idle-control")
c, args = v2.parse_args()
if idle:
    if not c.no_learn:
        raise ValueError("Idle control is evaluation only")
    original_table = v2.motor_table
    def idle_table(specs):
        table = original_table(specs)
        target = [0 if s.name == "shoot" else 1 for s in specs]
        index = table.index(target)
        v2.Learner.act = lambda self, state: index
        return table
    v2.motor_table = idle_table
v2.run(c, args.run_dir, args.env_path, args.port, args.warm_start)
