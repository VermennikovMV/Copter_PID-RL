#!/usr/bin/env python3
"""Read actual autopilot parameters from saved binary flight logs, read-only."""
import argparse
import json
from pathlib import Path
from pymavlink import mavutil
from verify_evidence import digest


def inspect(path):
    wanted={"AHRS_EKF_TYPE","EK3_ENABLE","EK2_ENABLE","SIM_RANDOM_SEED","SIM_SPEEDUP","GUID_OPTIONS",
            "MOT_THST_HOVER","MOT_HOVER_LEARN","MOT_THST_EXPO","ARMING_CHECK","FRAME_CLASS","FRAME_TYPE"}
    stream=mavutil.mavlink_connection(str(path))
    parameters={}; messages=[]
    while True:
        message=stream.recv_match(type=["PARM","MSG"])
        if message is None:break
        if message.get_type()=="MSG":
            if len(messages)<5:messages.append(str(message.Message))
        elif message.Name in wanted:
            parameters[message.Name]=float(message.Value)
    return {"path":str(path),"sha256":digest(path),"bytes":path.stat().st_size,"parameters":parameters,"startup_messages":messages}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    records=[]
    for path in sorted(args.root.rglob("*.BIN")):
        if path.name.lower()=="eeprom.bin":continue
        folder=path.parent.parent.parent
        if not (folder/"summary.json").exists():continue
        record=inspect(path)
        record["run"]=str(folder)
        records.append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps([{ "run":r["run"],"AHRS_EKF_TYPE":r["parameters"].get("AHRS_EKF_TYPE"),"SIM_RANDOM_SEED":r["parameters"].get("SIM_RANDOM_SEED")} for r in records],ensure_ascii=False))


if __name__=="__main__":main()
