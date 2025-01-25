# -*- coding: utf-8 -*- 
# @Time : 2024/9/19 15:03 
# @Author : DirtyBoy 
# @File : utils.py
import zipfile, json,hashlib
from datetime import datetime

def calculate_hashes(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_dex_timestamp(apk_path):
    try:
        with zipfile.ZipFile(apk_path, 'r') as apk:
            if 'classes.dex' in apk.namelist():
                dex_info = apk.getinfo('classes.dex')
                dex_timestamp = dex_info.date_time
                return datetime(*dex_timestamp)
            else:
                return None
    except Exception as e:
        return None

def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def read_vt_report(report_path):
    with open(report_path, 'r') as f:
        content = f.read()
        malicious = json.loads(content)['data']['attributes']['bundle_info']
    return malicious
