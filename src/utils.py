import os

def get_path_name(path):
    current_path = os.getcwd()
    package_name = "src"

    if package_name not in current_path:
        return f"{current_path}/{package_name}/{path}"
    else:
        return f"{current_path}/{path}"