"""Read-only PCAP clock evidence for the local Gazebo JSON + SITL MAVLink link.

Parses classic PCAP, Ethernet/Linux cooked loopback, UDP JSON and reassembled
server-to-client TCP MAVLink v1/v2. Only time-bearing attitude/position messages
are decoded, with their MAVLink CRC checked. No controller modules are imported.
"""
from __future__ import annotations
import bisect
import gzip
import json
import statistics
import struct
import re
from collections import Counter


def frames(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rb") as stream:
        header=stream.read(24)
        formats={b"\xd4\xc3\xb2\xa1":("<",1e-6),b"\xa1\xb2\xc3\xd4":(">",1e-6),
                 b"\x4d\x3c\xb2\xa1":("<",1e-9),b"\xa1\xb2\x3c\x4d":(">",1e-9)}
        if header[:4] not in formats:raise ValueError("Unsupported PCAP magic (classic PCAP required)")
        endian,scale=formats[header[:4]]
        linktype=struct.unpack(endian+"I",header[20:24])[0]
        while True:
            record=stream.read(16)
            if not record:break
            if len(record)!=16:raise ValueError("Truncated PCAP record header")
            sec,fraction,length,original=struct.unpack(endian+"IIII",record)
            if length>50_000_000:raise ValueError("Invalid PCAP capture length")
            packet=stream.read(length)
            if len(packet)!=length:raise ValueError("Truncated PCAP packet")
            yield sec+fraction*scale,linktype,packet


def ipv4(linktype,packet):
    if linktype==1: # Ethernet framing is also used by Linux loopback capture.
        if len(packet)<14:return None
        protocol=packet[12:14];offset=14
        if protocol==b"\x81\x00":protocol=packet[16:18];offset=18
        if protocol!=b"\x08\x00":return None
    elif linktype==113:
        if packet[14:16]!=b"\x08\x00":return None
        offset=16
    elif linktype==276:
        if packet[:2]!=b"\x08\x00":return None
        offset=20
    elif linktype in (101,228):offset=0
    elif linktype==0:offset=4
    else:raise ValueError("Unsupported capture linktype "+str(linktype))
    ip=packet[offset:]
    if len(ip)<20 or ip[0]>>4!=4:return None
    header_length=(ip[0]&15)*4
    total=struct.unpack("!H",ip[2:4])[0]
    if struct.unpack("!H",ip[6:8])[0]&0x3fff:return None # incomplete IP fragments
    return ip[9],ip[12:16],ip[16:20],ip[header_length:total]


def x25(data):
    crc=0xffff
    for byte in data:
        tmp=byte^(crc&255)
        tmp^=(tmp<<4)&255
        crc=((crc>>8)^(tmp<<8)^(tmp<<3)^(tmp>>4))&0xffff
    return crc


class MavClockStream:
    def __init__(self,targets=False):
        self.data=b"";self.next_seq=None;self.gaps=0;self.crc_bad=0;self.targets=targets

    def feed(self,seq,payload):
        if self.next_seq is not None:
            if seq<self.next_seq:
                payload=payload[self.next_seq-seq:]
                seq=self.next_seq
            elif seq>self.next_seq:
                self.gaps+=1;self.data=b""
        if not payload:return []
        self.next_seq=seq+len(payload)
        self.data+=payload
        decoded=[]
        while len(self.data)>=8:
            if self.data[0] not in (0xfe,0xfd):self.data=self.data[1:];continue
            version2=self.data[0]==0xfd
            header=10 if version2 else 6
            if len(self.data)<header:break
            size=self.data[1]
            total=header+size+2+(13 if version2 and self.data[2]&1 else 0)
            if len(self.data)<total:break
            message=self.data[:total];self.data=self.data[total:]
            msgid=int.from_bytes(message[7:10],"little") if version2 else message[5]
            if msgid not in ((82,) if self.targets else (30,32)):continue
            extra={30:39,32:185,82:49}[msgid]
            checksum=x25(message[1:header+size]+bytes([extra]))
            expected=int.from_bytes(message[header+size:header+size+2],"little")
            if checksum!=expected:self.crc_bad+=1;continue
            payload=message[header:header+size]+b"\0"*max(0,(39 if self.targets else 28)-size)
            boot=struct.unpack("<I",payload[:4])[0]*.001
            if self.targets:
                decoded.append((boot,*struct.unpack("<ffff",payload[20:36]),payload[38]))
            else:decoded.append((msgid,boot))
        return decoded


def clock_evidence(path,boot_start=None,boot_end=None,commands=None):
    udp_samples=[];mav_samples=[];target_samples=[];streams={};counts=Counter();time_sync_values=Counter()
    for timestamp,linktype,packet in frames(path):
        counts["packets"]+=1
        parsed=ipv4(linktype,packet)
        if parsed is None:continue
        protocol,source,dest,body=parsed
        if len(body)<8:continue
        sport,dport=struct.unpack("!HH",body[:4])
        if protocol==17:
            if sport not in (9002,9003) and dport not in (9002,9003):continue
            payload=body[8:]
            if b'"timestamp"' not in payload:continue
            try:value=json.loads(payload.strip(b"\x00\r\n "))
            except (ValueError,UnicodeDecodeError):continue
            if "timestamp" in value:
                udp_samples.append((timestamp,float(value["timestamp"])))
                time_sync_values[str(value.get("no_time_sync"))]+=1
                counts["fdm_json"]+=1
        elif protocol==6 and (sport==5760 or dport==5760):
            if len(body)<20:continue
            offset=(body[12]>>4)*4
            if offset<20:continue
            if len(body)<=offset:continue
            seq=struct.unpack("!I",body[4:8])[0]
            key=(source,dest,sport,dport)
            targets=dport==5760
            stream=streams.setdefault(key,MavClockStream(targets=targets))
            decoded=stream.feed(seq,body[offset:])
            if targets:
                target_samples.extend(decoded);counts["mav_attitude_commands_crc_valid"]+=len(decoded)
            else:
                for msgid,boot in decoded:
                    mav_samples.append((timestamp,boot,msgid));counts["mav_clock_crc_valid"]+=1
    udp_samples.sort()
    udp_wall=[row[0] for row in udp_samples]
    offsets=[]
    for wall,boot,msgid in mav_samples:
        if msgid!=32:continue
        if boot_start is not None and boot<boot_start:continue
        if boot_end is not None and boot>boot_end:continue
        index=bisect.bisect_right(udp_wall,wall)-1
        if index>=0 and wall-udp_wall[index]<.05:
            offsets.append(udp_samples[index][1]-boot)
    result={"path":str(path),"packet_counts":dict(counts),"fdm_no_time_sync_values":dict(time_sync_values),
            "tcp_sequence_gaps":sum(s.gaps for s in streams.values()),"mav_clock_crc_failures":sum(s.crc_bad for s in streams.values())}
    if offsets:
        median=statistics.median(offsets)
        ordered=sorted(abs(x-median) for x in offsets)
        result["clock_alignment"]={"gazebo_sim_minus_boot_s":median,
                                   "packet_alignment_abs_residual_p95_s":ordered[int(.95*(len(ordered)-1))],
                                   "pairs":len(offsets),"min_offset_s":min(offsets),"max_offset_s":max(offsets),
                                   "outliers_over_50ms":sum(abs(x-median)>.05 for x in offsets),
                                   "method":"kernel PCAP timestamps; preceding FDM JSON timestamp vs CRC-checked MAVLink LOCAL_POSITION_NED boot time"}
    result["capture_limit"]="TCP sequence gaps describe this passive capture/reassembly, not measured MAVLink packet loss; only CRC-valid complete messages contribute."
    capture_log=path.parent/"network_capture.log"
    if capture_log.exists():
        content=capture_log.read_text(errors="replace")
        result["tcpdump_counters"]={label:int(value) for value,label in re.findall(r"(\d+) packets (captured|received by filter|dropped by kernel)",content)}
    if commands and commands[0].get("action") is not None:
        cmd_times=[row["boot_s"] for row in commands];matches=0;checked=0;worst=0.;mask_bad=0
        for boot,p,q,r,thrust,mask in target_samples:
            if boot<cmd_times[0]-.002 or boot>cmd_times[-1]+.002:continue
            index=bisect.bisect_left(cmd_times,boot)
            candidates=[i for i in (index-1,index,index+1) if 0<=i<len(cmd_times)]
            index=min(candidates,key=lambda i:abs(cmd_times[i]-boot))
            if abs(cmd_times[index]-boot)>.002:continue
            command=commands[index];a=command["action"]
            expected=[2*a[1],-2*a[2],-a[3],command["guid_thrust"]]
            residual=max(abs(x-y) for x,y in zip((p,q,r,thrust),expected))
            checked+=1;matches+=residual<1e-6 and mask==128;mask_bad+=mask!=128;worst=max(worst,residual)
        result["independent_wire_commands"]={"checked":checked,"matched":matches,"max_difference":worst,"mask_mismatches":mask_bad,
                                              "status":"PASS" if checked>=10 and matches==checked else "FAIL",
                                              "scope":"CRC-valid incoming SET_ATTITUDE_TARGET captured on TCP, compared against independently replayed policy commands."}
    return result


if __name__=="__main__":
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument("path",type=Path)
    parser.add_argument("--boot-start",type=float)
    parser.add_argument("--boot-end",type=float)
    args=parser.parse_args()
    print(json.dumps(clock_evidence(args.path,args.boot_start,args.boot_end),indent=2))
