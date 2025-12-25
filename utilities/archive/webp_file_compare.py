import os
import random
import subprocess
import platform
import shutil
import time
from pathlib import Path

TEMP_DIR = Path.cwd() / "temp_display"

def get_directory(prompt_text):
    path = input(prompt_text).strip()
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Invalid directory: {path}")
    return Path(path)

def open_file(filepath):
    if platform.system() == "Darwin":
        subprocess.run(["open", filepath])
    elif platform.system() == "Windows":
        os.startfile(filepath)
    else:
        subprocess.run(["xdg-open", filepath])

def find_matching_dirs(parent_a, parent_b):
    dir_a_map = {d.name: d for d in parent_a.rglob('*') if d.is_dir()}
    dir_b_list = [d for d in parent_b.rglob('*') if d.is_dir()]
    matched_pairs = []

    for name_a, path_a in dir_a_map.items():
        key = name_a[:8]
        for path_b in dir_b_list:
            if path_b.name.startswith(key):
                matched_pairs.append((path_a, path_b))
                break
    return matched_pairs

def prepare_temp_dir():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

def copy_for_display(original_path, folder_label):
    safe_label = folder_label.replace(" ", "_").replace("[", "").replace("]", "")
    new_name = f"{safe_label}__{original_path.name}"
    temp_path = TEMP_DIR / new_name

    try:
        shutil.copy2(original_path, temp_path)
    except Exception as e:
        print(f"⚠️ Failed to copy {original_path.name}: {e}")

    return temp_path

def show_image_pairs(dir1, dir2):
    webp_files = [f.name for f in dir1.glob("*.webp")]
    if len(webp_files) < 10:
        print(f"⚠️  Skipping {dir1} — not enough .webp files ({len(webp_files)} found)")
        return None

    selected = random.sample(webp_files, 10)
    label1 = dir1.name
    label2 = dir2.name

    print(f"\n📂 Comparing: {label1} <--> {label2}")
    prepare_temp_dir()

    for i, filename in enumerate(selected, 1):
        print(f"\n  [{i}/10] {filename}")
        path1 = dir1 / filename
        path2 = dir2 / filename

        if not path1.exists():
            print(f"  ⚠️ Missing from A: {filename}")
            continue

        fake1 = copy_for_display(path1, label1)
        open_file(str(fake1))
        time.sleep(0.2)

        if path2.exists():
            fake2 = copy_for_display(path2, label2)
            open_file(str(fake2))
        else:
            print(f"  ⚠️ Missing from B: {filename}")

        if i < 10:
            input("Press Enter for next pair...")

    while True:
        print(f"\n🏁 Vote time:")
        print(f"   [a] {label1}  ({dir1})")
        print(f"   [b] {label2}  ({dir2})")
        print(f"   [skip] No clear winner")
        result = input("Your choice [a/b/skip]: ").strip().lower()

        if result in ['a', 'b', 'skip']:
            if result == 'a':
                return dir1.resolve()
            elif result == 'b':
                return dir2.resolve()
            else:
                return None
        else:
            print("Invalid input. Please type 'a', 'b', or 'skip'.")

def main():
    print("Recursive image pair viewer + renamed copy preview + winner tracker")
    parent_a = get_directory("Enter path to PARENT directory A: ")
    parent_b = get_directory("Enter path to PARENT directory B: ")

    matches = find_matching_dirs(parent_a, parent_b)
    if not matches:
        print("❌ No matching subdirectories found.")
        return

    print(f"\n🔍 Found {len(matches)} matching folder pairs.\n")

    winners = []

    for path_a, path_b in matches:
        winner = show_image_pairs(path_a, path_b)
        if winner:
            winners.append(winner)
        input("\n🔄 Press Enter to continue to the next matching folder...")

    print("\n🏆 SUMMARY OF WINNERS:")
    if winners:
        for w in winners:
            print("  -", w)
        with open("winners.txt", "w") as f:
            for w in winners:
                f.write(str(w) + "\n")
        print("\n📝 Saved to: winners.txt")
    else:
        print("  (No winners selected)")

if __name__ == "__main__":
    main()
