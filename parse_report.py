import json
from pathlib import Path

report_path = Path(__file__).parent / "report.json"
report = json.loads(report_path.read_text())

out_path = Path(__file__).parent / "output.json"


output:list[dict] = []

for data in report:
    temp = {
        "config": data["config"]["brain"]["params"],
        "result": data["report"]
    }
    output.append(temp)

d_out = json.dumps(output, indent=2)

out_path.write_text(d_out)