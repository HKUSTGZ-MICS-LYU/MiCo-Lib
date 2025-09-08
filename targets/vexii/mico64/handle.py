import os

replace = {
    "addi a5, a5, 4": "addi a5, a5, 8",
    "addi a1, a1, 4": "addi a1, a1, 8",
    "lw t2, 0(a5)": "ld t2, 0(a5)",
    "lw t3, 0(a1)": "ld t3, 0(a1)",
    "lw t3, 0(a5)": "ld t3, 0(a5)",
    "lw t2, 0(a1)": "ld t2, 0(a1)",
    "addi t1, t1, 32": "addi t1, t1, 64",
}

def handle(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    for old, new in replace.items():
        content = content.replace(old, new)

    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(current_dir):
        if filename.endswith(".S"):
            file_path = os.path.join(current_dir, filename)
            handle(file_path)