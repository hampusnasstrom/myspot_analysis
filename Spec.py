from datetime import datetime
from typing import List, Tuple
import json
import pandas as pd


class DateTimeEncoder(json.JSONEncoder):
    def default(self, z):
        if isinstance(z, datetime):
            return str(z)
        else:
            return super().default(z)


def read_spec(file_path: str) -> Tuple[dict, List[Tuple[dict, pd.DataFrame]]]:
    # https://certif.com/downloads/css_docs/spec_man.pdf
    with open(file_path, 'r') as file_handle:
        file_info = {
            "file_name": None,
            "epoch": None,
            "datetime": None,
            "motors": [],
            "comments": []
        }
        scans = []

        # Read file info
        line = file_handle.readline()
        while line != "\n":
            if line.startswith('#F'):
                file_info["file_name"] = line.split()[1]
            elif line.startswith('#E'):
                file_info["epoch"] = int(line.split()[1])
            elif line.startswith('#D'):
                file_info["datetime"] = datetime.strptime(line[3:27], "%a %b %d %H:%M:%S %Y")
            elif line.startswith("#O"):
                file_info["motors"] += line.split()[1:]
            elif line.startswith("#C"):
                comments = file_info["comments"]
                comments.append(line[3:-1])
                file_info["comments"] = comments
            else:
                raise ValueError("Not a valid spec file")
            line = file_handle.readline()

        # Read scans
        lines = file_handle.readlines()
        while "\n" in lines:
            scan_end = lines.index("\n")
            scan = lines[:scan_end]
            if len(lines) > scan_end + 1:
                lines = lines[scan_end + 1:]
            else:
                break
            scans.append(read_scan(scan))
        scans.append(read_scan(lines))
    return file_info, scans


def read_scan(lines: List[str]):
    scan_info = {
        "scan_number": None,
        "datetime": None,
        "motor_positions": [],
        "comments": []
    }
    columns = []
    scan_list = []
    for line in lines:
        if line.startswith('#'):
            if line.startswith('#S'):
                scan_info["scan_number"] = int(line.split()[1])
            elif line.startswith('#D'):
                scan_info["datetime"] = datetime.strptime(line, '#D %a %b %d %H:%M:%S %Y\n')
            elif line.startswith("#P"):
                scan_info["motor_positions"] += list(map(float, line.split()[1:]))
            elif line.startswith("#C"):
                comments = scan_info["comments"]
                comments.append(line[3:-1])
                scan_info["comments"] = comments
            elif line.startswith("#L"):
                columns = line.split()[1:]
        else:
            data = line.split()
            if len(data) == len(columns):
                scan_list.append(line.split())
    df = pd.DataFrame(data=scan_list, columns=columns)
    return scan_info, df


if __name__ == "__main__":
    eval_path = r"d:\Profile\oah\Eigene Dateien\210408_nyfs_kwz_bc_MAPI_NMP"
    eval_file_info, eval_scans = read_spec(eval_path)
    print(json.dumps(eval_file_info, indent=4, cls=DateTimeEncoder))
    for eval_scan in eval_scans:
        print(json.dumps(eval_scan[0], indent=4, cls=DateTimeEncoder))
        print(eval_scan[1])
