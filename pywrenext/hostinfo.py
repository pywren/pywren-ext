"""
Code to learn things about remote workers, like MAC address, 
NTP timing, etc. 
"""

import numpy as np
import time
import re
import subprocess
import ntplib

NTP_SERVERS = ['time.mit.edu', 
               'ntp1.net.berkeley.edu', 
               'ntp2.net.berkeley.edu']

def get_time_offset(server, attempts=1):
    """
    Returns a list of offsets for a particular server
    """
    import ntplib

    c = ntplib.NTPClient()

    res = []
    for i in range(attempts):
        try:
            r = c.request(server, version=3)
            offset = r.offset
            delay = r.delay
            res.append(offset)
        except ntplib.NTPException:
            pass
    return res

def parse_ifconfig_hwaddr(s):

    a = re.search(r'.+?(HWaddr\s+(?P<hardware_address>\S+))', s, re.MULTILINE).groupdict('')
    return a['hardware_address']

def parse_ifconfig_inetaddr(s):
    return re.findall(r'.+?inet addr:(?P<inet_addr>[\d.]+)', s, re.MULTILINE)

def get_hwaddr():
    ifconfig_data = subprocess.check_output("/sbin/ifconfig")
    hwaddr = parse_ifconfig_hwaddr(ifconfig_data)
    return hwaddr

def get_ifconfig():
    ifconfig_data = subprocess.check_output("/sbin/ifconfig")
    hwaddr = parse_ifconfig_hwaddr(ifconfig_data)
    inet_addr = parse_ifconfig_inetaddr(ifconfig_data)
    return hwaddr, inet_addr

def get_uptime():
    uptime_str = open("/proc/uptime").read().strip()
    up_str, idle_str = uptime_str.split()

    return float(up_str), float(idle_str)

