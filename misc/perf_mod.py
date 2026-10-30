import biotite
import os
import shutil
import sys
from datetime import datetime

def backup_file(file_path):
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        return backup_path
    return None

def copy_with_permissions(src, dst):
    shutil.copy2(src, dst)
    os.chmod(dst, 0o644)  # rw-r--r--


def replace_file(source_file, target_dir, target_file):
    try:

        backup_path = backup_file(target_file)
        if backup_path:
            print(f"create backup: {backup_path}")

        copy_with_permissions(source_file, target_file)

        if os.path.exists(target_file):
            file_size = os.path.getsize(target_file)
        else:
            raise FileNotFoundError("fail to replace file")

    except Exception as e:
        print(f"error: {str(e)}")
        if 'backup_path' in locals() and backup_path:
            try:
                shutil.copy2(backup_path, target_file)
            except Exception as restore_error:
                print(f"backup failed: {str(restore_error)}")
        sys.exit(1)

def main_bio():
    try:
        bio_path = os.path.dirname(biotite.__file__)
    except Exception as e:
        print(f"Could not find Biotite: {str(e)}")
        sys.exit(1)

    source_file = "misc/modify/convert.py"
    target_dir = os.path.join(bio_path, "structure/io/pdbx")
    target_file = os.path.join(target_dir, "convert.py")

    replace_file(source_file, target_dir, target_file)

    source_file = "misc/modify/file.py"
    target_dir = os.path.join(bio_path, "structure/io/pdb")
    target_file = os.path.join(target_dir, "file.py")

    replace_file(source_file, target_dir, target_file)


    source_file = "misc/modify/chains.py"
    target_dir = os.path.join(bio_path, "structure")
    target_file = os.path.join(target_dir, "chains.py")

    replace_file(source_file, target_dir, target_file)


    source_file = "misc/modify/resutil.py"
    target_dir = os.path.join(bio_path, "structure")
    target_file = os.path.join(target_dir, "resutil.py")

    replace_file(source_file, target_dir, target_file)





if __name__ == "__main__":
    main_bio()
