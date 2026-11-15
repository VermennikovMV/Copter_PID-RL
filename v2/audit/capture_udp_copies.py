"""Recognize Ethernet observations before/after UDP checksum completion.

This rule never reads application values, timestamps, or nearest neighbours.
Only a valid-checksum, reversed-MAC observation of an earlier invalid-checksum
datagram is suppressed. Every other IP byte, including ID, must match exactly.
The earlier observation remains in capture order. Pairing consumes one earlier
observation, so a later same-payload transmission cannot reuse it indefinitely.
"""
from collections import defaultdict, deque
import hashlib
import struct


def internet_checksum(data):
    if len(data) % 2:
        data += b'\x00'
    value = sum(struct.unpack('!' + 'H' * (len(data) // 2), data))
    while value >> 16:
        value = (value & 65535) + (value >> 16)
    return (~value) & 65535


def ethernet_ipv4_udp(packet, linktype):
    """Return exact bytes only for unfragmented Ethernet IPv4 UDP evidence."""
    if linktype != 1 or len(packet) < 14:
        return None
    offset = 14
    protocol = packet[12:14]
    if protocol == b'\x81\x00' and len(packet) >= 18:
        protocol, offset = packet[16:18], 18
    if protocol != b'\x08\x00' or len(packet) < offset + 20:
        return None
    ip = packet[offset:]
    ihl = (ip[0] & 15) * 4
    total = int.from_bytes(ip[2:4], 'big')
    if ip[0] >> 4 != 4 or ihl < 20 or total < ihl + 8 or len(ip) < total:
        return None
    ip = ip[:total]
    if ip[9] != 17 or int.from_bytes(ip[6:8], 'big') & 0x3fff:
        return None
    if internet_checksum(ip[:ihl]) != 0:
        return None
    udp = ip[ihl:]
    if int.from_bytes(udp[4:6], 'big') != len(udp):
        return None
    checksum = int.from_bytes(udp[6:8], 'big')
    # IPv4 zero means checksum disabled, not evidence of offload completion.
    if checksum == 0:
        return None
    valid = internet_checksum(ip[12:20] + b'\x00\x11' + udp[4:6] + udp) == 0
    check_offset = ihl + 6
    normalized = ip[:check_offset] + b'\x00\x00' + ip[check_offset + 2:]
    return ip, packet[:offset], normalized, valid, check_offset


class CaptureCopyFilter:
    def __init__(self):
        self.pending = defaultdict(deque)
        self.copies_removed = 0
        self.invalid_checksum_observations = 0
        self.valid_checksum_observations = 0
        self.unmatched_valid_observations = 0
        self.last_match = None
        self.examples = []

    def observe(self, packet, linktype, frame_index, wall):
        """True only for the later checksum-completed copy of an earlier frame."""
        self.last_match = None
        parsed = ethernet_ipv4_udp(packet, linktype)
        if parsed is None:
            return False
        ip, link, normalized, valid, checksum_offset = parsed
        source_mac, dest_mac, link_tail = link[6:12], link[:6], link[12:]
        if source_mac == dest_mac:
            return False
        key = (normalized, source_mac, dest_mac, link_tail)
        if not valid:
            self.invalid_checksum_observations += 1
            self.pending[key].append((frame_index, wall, ip, link))
            return False
        self.valid_checksum_observations += 1
        reverse_key = (normalized, dest_mac, source_mac, link_tail)
        queue = self.pending.get(reverse_key)
        if not queue:
            self.unmatched_valid_observations += 1
            return False
        first_index, first_wall, first_ip, first_link = queue.popleft()
        if not queue:
            del self.pending[reverse_key]
        # normalized bytes are the dictionary key: all other IP bytes match.
        self.copies_removed += 1
        self.last_match = {
            'first_capture_frame': first_index, 'later_capture_frame': frame_index,
            'first_wall_time': first_wall, 'later_wall_time': wall,
            'observed_copy_delay_s': wall - first_wall,
            'ipv4_id': int.from_bytes(ip[4:6], 'big'),
            'first_link_header_hex': first_link.hex(), 'later_link_header_hex': link.hex(),
            'first_ipv4_sha256': hashlib.sha256(first_ip).hexdigest(),
            'later_ipv4_sha256': hashlib.sha256(ip).hexdigest(),
            'ipv4_except_udp_checksum_sha256': hashlib.sha256(normalized).hexdigest(),
            'udp_checksum_byte_offsets': [checksum_offset, checksum_offset + 1],
            'differing_ipv4_byte_offsets': [i for i, (a, b) in enumerate(zip(first_ip, ip)) if a != b],
            'first_udp_checksum_valid': False, 'later_udp_checksum_valid': True,
        }
        if len(self.examples) < 8:
            self.examples.append(self.last_match)
        return True

    def diagnostics(self):
        return {'rule': 'Earlier invalid UDP checksum -> later valid checksum; reversed Ethernet MACs; every IPv4 byte including ID and complete UDP bytes identical except the two UDP checksum bytes. Consume each earlier observation once. First observation retained; no application values or timestamp window used.',
                'copies_removed': self.copies_removed,
                'invalid_checksum_observations': self.invalid_checksum_observations,
                'valid_checksum_observations': self.valid_checksum_observations,
                'unmatched_valid_observations': self.unmatched_valid_observations,
                'pending_invalid_observations': sum(len(q) for q in self.pending.values()),
                'examples': self.examples}
