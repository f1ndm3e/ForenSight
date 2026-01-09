#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
import time

# ================= CONFIG =================
"""
⚠️ USER CONFIG REQUIRED ⚠️

Before running this script:
1. Install VirtualBox + Guest Additions
2. Create a clean snapshot
3. Enable shared folder
4. Install 7-Zip in guest VM
5. Update ALL CONFIG values below
"""

# Path to VBoxManage binary
# macOS / Linux: usually just "VBoxManage"
# Windows: use full path to VBoxManage.exe
VBOXMANAGE = "<path>"

VM_NAME = "<your_vm_name>"
SNAPSHOT_NAME = "<>your_snapshot_name>"

GUEST_USER = "your_username"
GUEST_PASS = "your_password"

SRC_DIR = "path_to_your_zip_files"
SHARED_DIR = "path_to_your_shared_folder"
DUMP_DIR = "path_to_save_memory_dumps"

GUEST_SHARED_DIR = r"path_in_guest_shared_folder"
SEVEN_ZIP_PATH = r"path_to_7zip_in_guest"
ZIP_PASSWORD = "your_zip_password"
# =========================================

def run_cmd(cmd):
    print("[+] Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

def ensure_vm_powered_off():
    print("[*] Ensuring VM is powered off...")
    subprocess.run(
        [VBOXMANAGE, "controlvm", VM_NAME, "poweroff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    while True:
        info = subprocess.check_output(
            [VBOXMANAGE, "showvminfo", VM_NAME, "--machinereadable"]
        ).decode()
        if 'VMState="poweroff"' in info:
            print("[+] VM is powered off")
            break
        time.sleep(2)

def restore_snapshot():
    run_cmd([VBOXMANAGE, "snapshot", VM_NAME, "restore", SNAPSHOT_NAME])

def start_vm_gui():
    run_cmd([VBOXMANAGE, "startvm", VM_NAME, "--type", "gui"])

def stop_vm():
    run_cmd([VBOXMANAGE, "controlvm", VM_NAME, "acpipowerbutton"])

def unzip_in_guest(zip_name):
    zip_path = GUEST_SHARED_DIR + zip_name
    out_dir = GUEST_SHARED_DIR + os.path.splitext(zip_name)[0]

    cmd = [
        VBOXMANAGE, "guestcontrol", VM_NAME, "run",
        "--username", GUEST_USER,
        "--password", GUEST_PASS,
        "--exe", SEVEN_ZIP_PATH,
        "--",
        "x",
        zip_path,
        f"-p{ZIP_PASSWORD}",
        f"-o{out_dir}",
        "-y"
    ]
    run_cmd(cmd)

def dump_memory(sample):
    os.makedirs(DUMP_DIR, exist_ok=True)
    out = os.path.join(DUMP_DIR, sample + ".core")
    run_cmd([VBOXMANAGE, "debugvm", VM_NAME, "dumpvmcore", "--filename", out])
    print("[+] Memory dump saved:", out)

def cleanup_extracted_folder(sample):
    extracted_dir = os.path.join(SHARED_DIR, sample)
    if os.path.isdir(extracted_dir):
        shutil.rmtree(extracted_dir)
        print("[*] Extracted folder removed")

def main():
    zips = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".zip"))
    if not zips:
        print("[-] No ZIP files found")
        return

    for zip_name in zips:
        sample = os.path.splitext(zip_name)[0]
        src = os.path.join(SRC_DIR, zip_name)
        dst = os.path.join(SHARED_DIR, zip_name)

        print("\n" + "=" * 70)
        print(f"[+] Processing sample: {zip_name}")

        shutil.move(src, dst)
        print("[*] ZIP moved to shared folder")

        ensure_vm_powered_off()
        restore_snapshot()
        time.sleep(3)

        start_vm_gui()
        print("[*] Waiting for Windows boot...")
        time.sleep(30)

        unzip_in_guest(zip_name)
        print("[✔] Sample extracted")

        print("\n🔥 MANUAL STEP REQUIRED 🔥")
        print("1) Switch to VirtualBox GUI")
        print("2) Execute the sample MANUALLY")
        print("3) Do whatever interaction you need")
        print("4) When ready to dump memory → type 'n' + Enter here\n")

        user_input = input(">>> Continue to memory dump? (n): ").strip().lower()
        if user_input != "n":
            print("[!] Skipped by user")
            continue

        dump_memory(sample)

        cleanup_extracted_folder(sample)
        stop_vm()
        time.sleep(10)

        os.remove(dst)
        print("[✔] Finished sample:", zip_name)

    print("\n🔥 ALL SAMPLES PROCESSED 🔥")

if __name__ == "__main__":
    main()
