"""
Patches ESP-PPQ's helper.py to fix the infinite hang when exporting
large models. Run this ONCE before running quantize_for_esp32.py.

Run: python patch_espdl.py
"""
import esp_ppq
import os

helper = os.path.join(
    os.path.dirname(esp_ppq.__file__),
    'parser', 'espdl', 'helper.py'
)

print(f"Helper file: {helper}")
print(f"Exists: {os.path.exists(helper)}")

with open(helper, 'r', encoding='utf-8') as f:
    code = f.read()

# Show lines around the problematic area so we can see exactly what's there
print("\n--- Lines around 'chunks' (line 860-870 approx) ---")
lines = code.splitlines()
for i, line in enumerate(lines):
    if 'chunks' in line or 'repr(' in line.lower():
        start = max(0, i-2)
        end   = min(len(lines), i+3)
        for j in range(start, end):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1:4d}: {lines[j]}")
        print()

# Try all known variants of the problematic line
patches = [
    # Variant 1 — original
    (
        's += f"\\nvalue: {repr(chunks)}"',
        's += f"\\nvalue: [tensor shape {getattr(chunks, \'shape\', type(chunks))}]"'
    ),
    # Variant 2 — with different quotes
    (
        "s += f'\\nvalue: {repr(chunks)}'",
        "s += f'\\nvalue: [tensor shape {getattr(chunks, \"shape\", type(chunks))}]'"
    ),
    # Variant 3 — with escaped braces
    (
        's += f"\\nvalue: {repr(chunks)}}"',
        's += f"\\nvalue: [tensor shape {getattr(chunks, \'shape\', type(chunks))}]"'
    ),
]

patched = False
for old, new in patches:
    if old in code:
        code = code.replace(old, new)
        patched = True
        print(f"✓ Patched with variant: {repr(old[:50])}")
        break

if patched:
    with open(helper, 'w', encoding='utf-8') as f:
        f.write(code)
    print("\nSUCCESS — patch applied.")
    print("Now run: python quantize_for_esp32.py")
else:
    print("Could not find the exact line to patch.")
    print("Showing full printable_tensor_value function:")
    in_func = False
    for i, line in enumerate(lines):
        if 'def printable_tensor_value' in line:
            in_func = True
        if in_func:
            print(f"{i+1:4d}: {line}")
            if i > 0 and in_func and line.strip() == '' and i > 860:
                break
